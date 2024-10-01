import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm  # Import tqdm

# Directory to save images
IMAGE_DIR = "xkcd_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Function to download and save an image
def download_image(image_url, image_id):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        img.save(image_path, format="JPEG")
        print(f"Downloaded: {image_path}")
        return image_path
    except Exception as e:
        print(f"Error downloading image {image_id} from {image_url}: {e}")
        return None

# Download images from the dataset
from datasets import load_dataset
dataset = load_dataset("olivierdehaene/xkcd")

# Use tqdm for progress tracking
for item in tqdm(dataset['train'], desc="Downloading images"):
    image_url = item['image_url']
    image_id = item['id']
    download_image(image_url, image_id)
