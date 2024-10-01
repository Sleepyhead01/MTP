import torch
import gc
import pandas as pd
import requests
import os
from PIL import Image
import time
from openai import OpenAI # Assuming you're using OpenAI API, adjust if different
from tqdm import tqdm
from datasets import load_dataset

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://d2e3-54-70-184-189.ngrok-free.app/v1"


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

print(f"Using model: {model}")

# Define constants
OUTPUT_FILE = "pixtral_xkcd_results.tsv"
ERROR_LOG_FILE = "pixtral_error_log.txt"

# Load dataset and sample a small fraction
dataset = load_dataset("olivierdehaene/xkcd")

def get_image_from_filename(filename):
    """Load image from local file."""
    try:
        image_path = os.path.join('xkcd_images', filename)
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {filename}: {e}")
        return None


question = '''Analyze the comic and explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any). Answer in a single paragraph.'''

model_name = "pixtral"

# # Generate prompt for API
# def generate_prompt():
#     # entity_name = entity_name.replace("_", " ")
#     return f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

# Function to interact with API and handle retries and rate limiting
def api_inference_with_retry(prompt, image_url):
    retries = 5
    retry_delay = 2
    for attempt in range(retries):
        try:
            chat_completion_from_base64 = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]}
                ],
                model=model,  # Replace with actual API model name
                max_tokens=256
            )
            return chat_completion_from_base64.choices[0].message.content           
        except Exception as e:
            print(f"API call failed: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff for rate limit errors
    return None

# Function to run batch inference and save results
def main():
    output_file = "pixtral_xkcd_results.tsv"

    # batch_size = 16  # Define batch size

    with open(output_file, 'w') as f:
        f.write("index\timage_url\tresponse\n")
    
    for item in tqdm(dataset['train'], desc="Processing items"):
        prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        # image = get_image_from_filename(str(item['id'])+".jpg")

        image_url = item['image_url']

        with open(output_file, 'a') as f:
            result = api_inference_with_retry(prompt, image_url)
            if result:
                f.write(f"{item['id']}\t{item['image_url']}\t{result}\n")
                f.flush()

if __name__ == "__main__":
    main()
