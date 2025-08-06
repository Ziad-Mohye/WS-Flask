from flask import Flask, request, render_template
import os
import tifffile as tiff
import numpy as np
from PIL import Image
import io
import base64
import torch
import segmentation_models_pytorch as smp

app = Flask(__name__)

# Path to your saved pytorch state_dict
MODEL_PATH = r'E:/Ziad/Cellula Internship/Fourth Task/Application by Flask/model_deeplabv3plus.pth' 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same model architecture used in training
def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name='resnet34',
        encoder_weights=None,  # weights none since we load custom weights
        in_channels=12,
        classes=1,
        decoder_dropout=0.5
    )
    return model

# Load model weights
model = build_model().to(device)
state = torch.load(MODEL_PATH, map_location=device)
# If you saved full model vs state_dict, handle accordingly:
if isinstance(state, dict) and 'state_dict' in state:
    model.load_state_dict(state['state_dict'])
else:
    model.load_state_dict(state)
model.eval()

def normalize_image(img):
    # img shape: (bands, H, W)
    img = img.astype(np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        # ignore NaNs
        if np.isnan(band).all():
            img[i] = 0.0
            continue
        min_val = np.nanmin(band)
        max_val = np.nanmax(band)
        if max_val > min_val:
            img[i] = (band - min_val) / (max_val - min_val)
        else:
            img[i] = 0.0
    return img

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)
    return file_path

def create_rgb_image(image_data):
    """
    image_data: numpy array shape (bands, H, W) or (H, W, bands)
    We'll accept both; use the first 3 bands (0,1,2) for RGB preview.
    """
    # ensure shape (bands, H, W)
    if image_data.ndim == 3 and image_data.shape[2] <= 12:
        # probably (H, W, bands)
        img = np.transpose(image_data, (2, 0, 1))
    else:
        img = image_data.copy()

    # Use first 3 bands (or duplicate last band if less than 3)
    bands = []
    for i in range(3):
        if i < img.shape[0]:
            channel = img[i]
        else:
            channel = img[-1]
        # normalize each channel
        ch = channel.astype(np.float32)
        cmin, cmax = np.nanmin(ch), np.nanmax(ch)
        if cmax > cmin:
            ch = (ch - cmin) / (cmax - cmin)
        else:
            ch = np.zeros_like(ch)
        bands.append(ch)
    rgb = np.stack(bands, axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

def generate_overlay(rgb_image, mask):
    """
    rgb_image: HxWx3 uint8
    mask: HxW binary (0/1) uint8
    returns a PIL Image blended overlay with mask in red
    """
    rgb_pil = Image.fromarray(rgb_image)
    # create mask image red channel
    mask_rgb = np.zeros_like(rgb_image)
    mask_rgb[:, :, 0] = (mask * 255).astype(np.uint8)  # red channel
    mask_pil = Image.fromarray(mask_rgb)
    # blend
    overlay = Image.blend(rgb_pil.convert('RGBA'), mask_pil.convert('RGBA'), alpha=0.4)
    return overlay.convert('RGB')

def convert_image_to_base64(pil_image):
    img_io = io.BytesIO()
    pil_image.save(img_io, 'PNG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return 'No selected file', 400

    file_path = save_uploaded_file(uploaded_file)

    # read tif (expects multi-band, e.g., 12 bands)
    image_data = tiff.imread(file_path)  # may be shape (bands, H, W) or (H, W, bands)

    # normalize channels and ensure shape (bands, H, W)
    if image_data.ndim == 3 and image_data.shape[2] <= 12:
        image_data = np.transpose(image_data, (2, 0, 1))

    image_data = normalize_image(image_data)  # (bands, H, W)

    # Prepare tensor for model: (1, C, H, W)
    input_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)  # shape (1, 1, H, W)
        preds = torch.sigmoid(preds)
        preds_np = preds.cpu().numpy()[0, 0]  # H, W

    mask = (preds_np > 0.5).astype(np.uint8)

    # create RGB preview from original image (use first 3 bands)
    # pass original (unnormalized) image for visualization - read it again in HWC form if needed
    # If original tiff read earlier was (bands,H,W), convert to HWC for display:
    if image_data.shape[0] >= 3:
        # image_data currently normalized float 0..1; create rgb using normalized values
        rgb_image = create_rgb_image(np.transpose(image_data, (1, 2, 0)))
    else:
        # fallback: replicate single band
        single = image_data[0]
        rgb_image = np.stack([single, single, single], axis=-1)
        rgb_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)

    overlay_image = generate_overlay(rgb_image, mask)

    encoded_input_image = convert_image_to_base64(Image.fromarray(rgb_image))
    encoded_overlay_image = convert_image_to_base64(overlay_image)

    return render_template('index.html', rgb_image=encoded_input_image, output_image=encoded_overlay_image)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
