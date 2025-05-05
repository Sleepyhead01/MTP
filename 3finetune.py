# import torch
# import gc
# import pandas as pd
# import requests
# import os
# from PIL import Image
# import time
# import json
# from openai import OpenAI
# from tqdm import tqdm
# from datasets import load_dataset

# # Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# # openai_api_base = "https://a534-54-70-184-189.ngrok-free.app/v1"  # llava
# # openai_api_base = "https://a47d-54-70-184-189.ngrok-free.app/v1" #mixtral/pixtral/
# openai_api_base = "https://2889-54-70-184-189.ngrok-free.app/v1" #microsoft/phi

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# models = client.models.list()
# model = models.data[0].id
# model_name = model.split("/")[0]
# print(f"Using model: {model_name}")

# # gen_type = "with_transcript"
# gen_type = "with_CoT"

# # Define constants
# OUTPUT_FILE = f"{model_name}_{gen_type}_xkcd_results.jsonl"
# ERROR_LOG_FILE = f"{model_name}_{gen_type}_error_log.txt"

# # Load dataset
# dataset = load_dataset("olivierdehaene/xkcd")

# question_template = '''
# Analyze the comic by providing a step-by-step breakdown of its elements and explain how they contribute to the humor or underlying message. In your analysis, consider any cultural, scientific, or technical references. Conclude with a **Final Explanation** summarizing the main points. Please present your answer in the following format to facilitate extraction of the final explanation:

# ```
# Step-by-Step Analysis:
# 1. ...
# 2. ...
# 3. ...
# ...

# Final Explanation:
# [Your summary here.]
# ```
# '''


# # Function to interact with API and handle retries and rate limiting
# def api_inference_with_retry(prompt, image_url):
#     retries = 5
#     retry_delay = 2
#     for attempt in range(retries):
#         try:
#             # Call the API with the prompt and image URL
#             chat_completion = client.chat.completions.create(
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": prompt},
#                             {"type": "image_url", "image_url": {"url": image_url}},
#                         ]
#                     }
#                 ],
#                 model=model,
#                 max_tokens=800
#             )
#             return chat_completion.choices[0].message.content
#         except Exception as e:
#             print(f"API call failed: {e}")
#             time.sleep(retry_delay)
#             retry_delay *= 2  # Exponential backoff for rate limit errors
#     return None

# # Function to run batch inference and save results
# def main():
#     output_file = f"{model_name}_{gen_type}_xkcd_results_ngrok_.jsonl"

#     with open(output_file, 'w', encoding='utf-8') as f:
#         for item in tqdm(dataset['train'], desc="Processing items"):
#             # Format the prompt
#             prompt = question_template

#             image_url = item['image_url']

#             result = api_inference_with_retry(prompt, image_url)
#             if result:
#                 # Create a dictionary for the result
#                 data = {
#                     "index": item['id'],
#                     "image_url": image_url,
#                     "response": result
#                 }
#                 # Write the JSON object to the file
#                 f.write(json.dumps(data) + '\n')
#                 f.flush()
#             else:
#                 # Log errors if needed
#                 with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as error_log:
#                     error_log.write(f"Failed to get response for item ID {item['id']}\n")

# if __name__ == "__main__":
#     main()



import torch
import gc
import pandas as pd
import requests
import os
from PIL import Image, ImageDraw, ImageFont
import time
import json
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
# openai_api_base = "https://a534-54-70-184-189.ngrok-free.app/v1"  # llava
# openai_api_base = "https://a47d-54-70-184-189.ngrok-free.app/v1" #microsoft/phi
openai_api_base = "https://2889-54-70-184-189.ngrok-free.app/v1" #mixtral/pixtral/

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
model_name = model.split("/")[0]
print(f"Using model: {model_name}")

# gen_type = "with_transcript"
gen_type = "1shot"


# Load dataset
dataset = load_dataset("olivierdehaene/xkcd")
ERROR_LOG_FILE = f"{model_name}_{gen_type}_error_log.txt"


# # Function to add a label below an image
# def add_label_below_image(image, label, font):
#     # Create a drawing context
#     draw = ImageDraw.Draw(image)
    
#     # Get the size of the text using textbbox
#     text_bbox = draw.textbbox((0, 0), label, font=font)
#     text_width = text_bbox[2] - text_bbox[0]
#     text_height = text_bbox[3] - text_bbox[1]
    
#     # Create a new image with extra space for the label
#     result = Image.new('RGB', (image.width, image.height + text_height + 10), (255, 255, 255))
#     result.paste(image, (0, 0))
    
#     # Draw the label on the new image
#     draw = ImageDraw.Draw(result)
#     text_x = (result.width - text_width) // 2
#     text_y = image.height + 5
#     draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
    
#     return result

# # Function to stack two images vertically with labels and increased font size
# def stack_images_with_labels(image1_url, image2_url):
#     # Load images from URLs
#     # image1 = Image.open(requests.get(image1_url, stream=True).raw).convert("RGB")
#     # image2 = Image.open(requests.get(image2_url, stream=True).raw).convert("RGB")
#     image1 = Image.open(image1_url).convert("RGB")
#     image2 = Image.open(image2_url).convert("RGB")

    
#     # Load a font with increased size
#     font_size = 20  # You can adjust this value as needed
#     try:
#         label_font = ImageFont.truetype("/home/satyanshu/Documents/testing/testing1/Arial.ttf", size=font_size)
#     except IOError:
#         # If "arial.ttf" is not available, use the default font
#         label_font = ImageFont.load_default()
#         print("Warning: 'arial.ttf' not found. Using default font.")
    
#     label1 = "EXAMPLE IMAGE"
#     label2 = "TEST IMAGE"
    
#     # Add labels below images
#     image1_with_label = add_label_below_image(image1, label1, label_font)
#     image2_with_label = add_label_below_image(image2, label2, label_font)
    
#     # Stack images vertically
#     total_height = image1_with_label.height + image2_with_label.height
#     max_width = max(image1_with_label.width, image2_with_label.width)
#     combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
#     combined_image.paste(image1_with_label, (0, 0))
#     combined_image.paste(image2_with_label, (0, image1_with_label.height))
    
#     # Save the combined image to a temporary file
#     combined_image_path = 'combined_image.jpg'
#     combined_image.save(combined_image_path)
#     return combined_image_path


# Function to interact with API and handle retries and rate limiting
def api_inference_with_retry(prompt, combined_image_path):
    retries = 5
    retry_delay = 2
    for attempt in range(retries):
        try:
            # Upload the combined image to a hosting service to get a URL
            # For demonstration purposes, we'll assume the image is accessible locally
            # In practice, you would upload the image to a server or cloud storage
            # image_url = os.path.abspath(combined_image_path)

            # Call the API with the prompt and image URL
            # Encode image
            # image_url = image_to_base64(image_url)
            image_url = combined_image_path

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                    }
                ],
                model=model,
                max_tokens=800
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff for rate limit errors
    return None

# Function to run batch inference and save results
def main():
    output_file = f"{model_name}_{gen_type}_xkcd_results_ngrok_.jsonl"

    # Use the first image and its explanation as the example
    # example_item = dataset['train'][0]
    # example_image_url = "https://www.explainxkcd.com/wiki/images/5/5f/rnaworld_2x.png"
    # example_image_url = "/home/satyanshu/Documents/testing/testing1/rnaworld_2x.png"

    example_explanation = """This comic humorously combines biology with Disney themes. It references the RNA world hypothesis, a theory on the origin of life where RNA serves both as genetic material and as a catalyst for its own replication.
    Ariel (The Little Mermaid) is depicted collecting nucleotides instead of human artifacts, aligning with RNA and DNA building blocks.
    Ratatouille, about a rat chef, is tied to "primordial soup," the early Earth's chemical environment where life began.
    Elsa (Frozen), who brings a snowman to life with her ice powers, is linked to the emergence of life through ribozyme synthesis—RNA molecules that catalyze biochemical reactions.
    The title text expands on this, suggesting Elsa's magic relies on ribozymes, making Olaf's life machinery RNA-based, consistent with the RNA World theme. It playfully notes Olaf, like cometary ice thought to deliver organic compounds to Earth, might be infused with carbon-rich molecules. The mention of "canonically" references official Disney lore."""

    # test_images_directory = "/home/satyanshu/Documents/testing/testing1/xkcd_images"  # Replace with the actual path

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset['train'], desc="Processing items"):
            # test_image_url = item['image_url']
            # test_image_filename = f"{item['id']}.jpg"
            # test_image_url = os.path.join(test_images_directory, test_image_filename)

            # Stack example image and test image
            # combined_image_path = stack_images_with_labels(example_image_url, test_image_url)
            combined_image_path = f"https://xkcdmmd.s3.ap-south-1.amazonaws.com/{item['id']}_stacked.jpg"

            # Format the prompt
            prompt = f"""The image contains an example along with its explanation below: \n```{example_explanation}``` \n### TASK ###\nPlease analyze the following test image and provide a detailed explanation in the same style as the example. Eexplain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any). """
            # from IPython.display import display, Image
            # display(Image(filename=combined_image_path))
            # print(prompt)
            # break
        
            result = api_inference_with_retry(prompt, combined_image_path)
            if result:
                # Create a dictionary for the result
                data = {
                    "index": item['id'],
                    # "example_image_url": example_image_url,
                    "test_image_url": combined_image_path,
                    "response": result
                }
                # Write the JSON object to the file
                f.write(json.dumps(data) + '\n')
                f.flush()
            else:
                # Log errors if needed
                with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as error_log:
                    error_log.write(f"Failed to get response for item ID {item['id']}\n")

if __name__ == "__main__":
    main()

#############################CREATE STACKED IMAGES ##########################################################################

# import os
# from PIL import Image, ImageDraw, ImageFont
# from tqdm import tqdm
# from datasets import load_dataset

# # Load dataset
# dataset = load_dataset("olivierdehaene/xkcd")

# # Function to add a label below an image
# def add_label_below_image(image, label, font):
#     # Create a drawing context
#     draw = ImageDraw.Draw(image)
    
#     # Get the size of the text using textbbox
#     text_bbox = draw.textbbox((0, 0), label, font=font)
#     text_width = text_bbox[2] - text_bbox[0]
#     text_height = text_bbox[3] - text_bbox[1]
    
#     # Create a new image with extra space for the label
#     result = Image.new('RGB', (image.width, image.height + text_height + 10), (255, 255, 255))
#     result.paste(image, (0, 0))
    
#     # Draw the label on the new image
#     draw = ImageDraw.Draw(result)
#     text_x = (result.width - text_width) // 2
#     text_y = image.height + 5
#     draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
    
#     return result

# # Function to stack two images vertically with labels and increased font size
# def stack_images_with_labels(image1_path, image2_path, output_path, label1="EXAMPLE IMAGE", label2="TEST IMAGE"):
#     # Load images from file paths
#     image1 = Image.open(image1_path).convert("RGB")
#     image2 = Image.open(image2_path).convert("RGB")

#     # Load a font with increased size
#     font_size = 20  # You can adjust this value as needed
#     try:
#         label_font = ImageFont.truetype("/home/satyanshu/Documents/testing/testing1/Arial.ttf", size=font_size)
#     except IOError:
#         # If "arial.ttf" is not available, use the default font
#         label_font = ImageFont.load_default()
#         print("Warning: 'arial.ttf' not found. Using default font.")
    
#     # Add labels below images
#     image1_with_label = add_label_below_image(image1, label1, label_font)
#     image2_with_label = add_label_below_image(image2, label2, label_font)
    
#     # Stack images vertically
#     total_height = image1_with_label.height + image2_with_label.height
#     max_width = max(image1_with_label.width, image2_with_label.width)
#     combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
#     combined_image.paste(image1_with_label, (0, 0))
#     combined_image.paste(image2_with_label, (0, image1_with_label.height))
    
#     # Save the combined image
#     combined_image.save(output_path)

# # Main function to process and save stacked images
# def main():
#     example_image_path = "/home/satyanshu/Documents/testing/testing1/rnaworld_2x.png"  # Path to example image
#     test_images_directory = "/home/satyanshu/Documents/testing/testing1/xkcd_images"  # Directory containing test images
#     output_directory = "/home/satyanshu/Documents/testing/testing1/stacked_images_"  # Directory to save stacked images

#     os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist

#     for item in tqdm(dataset['train'], desc="Processing items"):
#         test_image_filename = f"{item['id']}.jpg"
#         test_image_path = os.path.join(test_images_directory, test_image_filename)

#         # Output file path for the stacked image
#         output_path = os.path.join(output_directory, f"{item['id']}_stacked.jpg")
        
#         # Check if the test image exists
#         if os.path.exists(test_image_path):
#             # Stack example image and test image
#             stack_images_with_labels(example_image_path, test_image_path, output_path)
#         else:
#             print(f"Test image not found: {test_image_path}")

# if __name__ == "__main__":
#     main()
