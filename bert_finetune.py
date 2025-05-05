# import torch
# import gc
# import pandas as pd
# import requests
# import os
# from PIL import Image
# import time
# from huggingface_hub import InferenceClient
# from tqdm import tqdm
# from datasets import load_dataset

# # Hugging Face API Key (replace with your actual key)
# hf_api_key = "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"
# client = InferenceClient(api_key=hf_api_key)

# # Define constants
# OUTPUT_FILE = "huggingface_xkcd_results.tsv"
# ERROR_LOG_FILE = "huggingface_error_log.txt"
# EXAMPLE_IMAGE_URL = "https://www.explainxkcd.com/wiki/images/a/ab/geometriphylogenetics_2x.png"  # URL for the example image
# DEBUG_MODE = True  # Set to True for testing with a single inference

# # Load dataset
# dataset = load_dataset("olivierdehaene/xkcd")

# # Define the few-shot prompt template with placeholders for images and transcript
# few_shot_prompt = '''
# <|user|>
# <|image_1|>
# Analyze the comic and transcript and finally explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any). Answer in a single paragraph.<|end|>
# <|assistant|>
# Phylogenetics refers to the practice of examining relationships among things that follow the principle of "descent with modification of progeny". In the course of descent with modification, one thing may give rise to two (the progeny), different modifications happen to each, and those modifications become established. Iterated "splits" over time yield a tree of objects; it is the purpose of phylogenetics to recover ("reconstruct") these trees, and use the information gained to inform study of the things contained. Phylogenetics has been most commonly applied to the classification/taxonomy of biological species and investigations of their evolutionary history, but it has also been used to examine the evolution of genes and biosynthetic pathways. Similar conceptual approaches have been used in the study of human languages and their evolution. 
# Data for phylogenetic analyses may come from any attributes ("characters") of the things being examined. Rigorous techniques for these analyses became available starting in the 1950s, and these replaced earlier methods based largely on the individual judgement of experts. In phylogenetic studies of organisms, their DNA is far and away the most data-dense source of information, and consequently, most present-day investigations are based on analyses of selected genes and, increasingly, whole genomes. Thanks to the advent of more robust datasets, and more robust methods of data analysis, it is now commonplace for studies, especially on relatively understudied creatures, to reconstruct an evolutionary history (a phylogeny) that is radically different from what had previously been assumed. This is the "phylogenetic revolution" referred to in the caption. One example is the genus Hippopotamus, which had been considered a relative of pigs, which the animals somewhat resemble, until modern data and methods revealed it to be more closely related to whales, despite the animals being very different physically (hippos spend time in water, but can't swim). 
# This comic presents a tree, which resembles and purports to be a phylogenetic tree, in which the endpoints ("terminal taxa") are geometric shapes. This has been given the name "geometriphylogenetics" — a portmanteau of "geometry" and "phylogenetics". The claim is that triangles have been moved from the lower part of the tree (where they would be closely related to squares, rectangles, pentangles, and the like) to the upper part (placing them closer to circles and ellipses). This is a riff on the findings, and even the wording, of authentic phylogenetic research papers that report "revolutionary" results. The absurdity, and the joke, is that geometries do not change over time. Human understanding might change, but triangles and circles and rectangles themselves did not evolve or descend from ancestors, they are not progeny of some other geometries of ages past, and therefore phylogenetic principles and techniques don't apply to their study. Moreover, geometries do not contain DNA,[citation needed] so genetic analysis, even if it was ever relevant, is impossible. 
# The title text alludes to maximum likelihood, one of the most robust, and most frequently used, methodologies for phylogenetic analysis. The method builds a number of trees from the data, assigns to each a probability that it conforms to a pre-selected model of evolution, and then selects the tree that has the highest (maximum) likelihood of conformity to the model. In this case, though, the statement "There's a [high probability] that I'm doing phylogenetics wrong" doesn't just have the maximum probability of the available options; it has the maximum possible probability of 1, because it is definitely the case.<|end|>
# <|user|>
# <|image_2|>
# Analyze the comic and transcript and finally explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any). Answer in a single paragraph.<|end|>
# <|assistant|>
# '''

# # Function to interact with API and handle retries and rate limiting
# def api_inference_with_retry(prompt, image_urls):
#     retries = 5
#     retry_delay = 2
#     for attempt in range(retries):
#         try:
#             # Call the Hugging Face Inference API with the prompt and the first image URL
#             response = client.text_completion(
#                 model="microsoft/Phi-3.5-vision-instruct",
#                 prompt=prompt,
                
#             )
#             return response
        
#         except Exception as e:
#             print(f"API call failed: {e}")
#             time.sleep(retry_delay)
#             retry_delay *= 2  # Exponential backoff for rate limit errors
#     return None

# # Function to run batch inference and save results
# def main():
#     if DEBUG_MODE:
#         # Run a single inference for debugging
#         item = dataset['train'][0]
#         prompt = few_shot_prompt
#         image_urls = [EXAMPLE_IMAGE_URL, item['image_url']]

#         result = api_inference_with_retry(prompt, image_urls)
#         if result:
#             print(f"Index: {item['id']}")
#             print(f"Image URLs: {', '.join(image_urls)}")
#             print(f"Response: {result}")
#     else:
#         with open(OUTPUT_FILE, 'w') as f:
#             f.write("index	image_urls	response")
        
#         # Run batch inference for all items in the dataset
#         for i in range(len(dataset['train'])):
#             # Get one item from the dataset
#             item = dataset['train'][i]

#             # Format the prompt with the few-shot example and the example image
#             prompt = few_shot_prompt
#             image_urls = [EXAMPLE_IMAGE_URL, item['image_url']]

#             with open(OUTPUT_FILE, 'a') as f:
#                 result = api_inference_with_retry(prompt, image_urls)
#                 if result:
#                     f.write(f"{item['id']}	{','.join(image_urls)}	{result}")
#                     f.flush()

# if __name__ == "__main__":
#     main()






# # import torch
# # print("CUDA available:", torch.cuda.is_available())
# # print("CUDA version:", torch.version.cuda)





# # sh cuda_12.6.3_560.35.05_linux.run --silent --toolkit --toolkitpath=$HOME/cuda

# # echo 'export PATH=$HOME/cuda/bin:$PATH' >> ~/.bashrc
# # echo 'export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# # source ~/.bashrc


# import requests
# from io import BytesIO
# from PIL import Image
# from torchvision.transforms import Resize
# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# # Model and processor setup
# model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

# # Quantization configuration with updated compute dtype
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     quantization_config=quant_config,
#     attn_implementation="flash_attention_2"
# )

# processor = AutoProcessor.from_pretrained(model_id)

# # Resize and validate utility
# resize_transform = Resize((224, 224))

# def validate_image(image):
#     if not image.mode == "RGB":
#         image = image.convert("RGB")
#     return image

# def fetch_image(url):
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     if "image" not in response.headers.get("content-type", ""):
#         raise ValueError(f"URL {url} does not point to an image.")
#     image = Image.open(BytesIO(response.content))
#     return validate_image(image)

# # Load and validate images
# image_urls = [
#     "https://www.explainxkcd.com/wiki/images/thumb/4/4a/sky_alarm_2x.png/331px-sky_alarm_2x.png",
#     "https://www.explainxkcd.com/wiki/images/thumb/4/4a/sky_alarm_2x.png/331px-sky_alarm_2x.png"
# ]
# raw_images = [resize_transform(fetch_image(url)) for url in image_urls]

# # Debug image sizes
# for idx, img in enumerate(raw_images):
#     print(f"Image {idx + 1} size: {img.size}, mode: {img.mode}")

# # Define interleaved conversation
# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Compare these two images and explain."},
#             {"type": "image"},
#             {"type": "text", "text": "What stands out between them?"},
#             {"type": "image"}
#         ],
#     },
# ]

# # Format prompt
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# # Prepare inputs
# inputs = processor(
#     images=raw_images,
#     text=prompt,
#     return_tensors="pt"
# ).to(0, torch.float16)

# # Generate output
# output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))



import requests
from io import BytesIO
from PIL import Image
from torchvision.transforms import Resize
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import time

DEBUG_MODE = False  # Set to True for testing with a single inference

# Model and processor setup
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

# Quantization configuration with updated compute dtype
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
    # quantization_config=quant_config,
    attn_implementation="flash_attention_2"
)

model = model.to('cuda:0')
processor = AutoProcessor.from_pretrained(model_id)

# Resize and validate utility
resize_transform = Resize((224, 224))

def validate_image(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image

def fetch_image(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    if "image" not in response.headers.get("content-type", ""):
        raise ValueError(f"URL {url} does not point to an image.")
    image = Image.open(BytesIO(response.content))
    return validate_image(image)

# # Construct Llava-compatible prompt
# def construct_prompt(example_urls, test_urls):
#     example_prompt = "<|im_start|>user <image><image>\nAnalyze these comics and explain their humor or underlying concept.|im_end|><|im_start|>assistant\n"
#     example_prompt += (
#         "These two images depict humor based on cultural and scientific references. The first comic highlights phylogenetic principles, while the second one explores geometric humor with a technical twist.|im_end|>"
#     )
#     test_prompt = "<|im_start|>user <image><image>\nAnalyze these comics and explain their humor or underlying concept.|im_end|><|im_start|>assistant"
#     return example_prompt, test_prompt

# Perform Llava inference
def llava_inference_with_retry(prompt, raw_images):
    retries = 5
    retry_delay = 2
    for attempt in range(retries):
        try:
            # Prepare inputs
            inputs = processor(
                images=raw_images,
                text=prompt,
                return_tensors="pt"
            ).to('cuda:0', torch.bfloat16)
            # .to(0, torch.float16)

            # Generate output
            output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            response = processor.decode(output[0][2:], skip_special_tokens=True)
            return response

        except Exception as e:
            print(f"Inference failed: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff for retry
    return None

# Load dataset
dataset = load_dataset("olivierdehaene/xkcd")

import json

# Constants
OUTPUT_FILE = "llava_xkcd_results_fewshot_bf16.jsonl"

# Main function to run inference
def main():
    if DEBUG_MODE:
        # Debugging with one example image and one test image
        example_url = "https://www.explainxkcd.com/wiki/images/5/5f/rnaworld_2x.png"
        test_url = dataset['train'][0]['image_url']

        # Fetch and process images
        example_image = resize_transform(fetch_image(example_url))
        test_image = resize_transform(fetch_image(test_url))

        # Construct prompt for one example and one test image
        example_prompt = (
            "<|im_start|>user <image>\nAnalyze the comic and transcript and finally explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any).|im_end|>"
            "<|im_start|>assistant\n"
            '''This comic humorously combines biology with Disney themes. It references the RNA world hypothesis, a theory on the origin of life where RNA serves both as genetic material and as a catalyst for its own replication.
Ariel (The Little Mermaid) is depicted collecting nucleotides instead of human artifacts, aligning with RNA and DNA building blocks.
Ratatouille, about a rat chef, is tied to "primordial soup," the early Earth's chemical environment where life began.
Elsa (Frozen), who brings a snowman to life with her ice powers, is linked to the emergence of life through ribozyme synthesis—RNA molecules that catalyze biochemical reactions.
The title text expands on this, suggesting Elsa's magic relies on ribozymes, making Olaf's life machinery RNA-based, consistent with the RNA World theme. It playfully notes Olaf, like cometary ice thought to deliver organic compounds to Earth, might be infused with carbon-rich molecules. The mention of "canonically" references official Disney lore.
'''
            "|im_end|>"
        )
        test_prompt = (
            "<|im_start|>user <image>\nAnalyze the comic and transcript and finally explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any).|im_end|>"
            "<|im_start|>assistant"
        )
        full_prompt = example_prompt + test_prompt

        # Perform inference with the model
        result = llava_inference_with_retry(full_prompt, [example_image, test_image])
        if result:
            print(f"Example Image URL: {example_url}")
            print(f"Test Image URL: {test_url}")
            print(f"Generated Response: {result}")

    else:
        # Batch inference for all items in the dataset
        with open(OUTPUT_FILE, 'w') as f:
            for i in tqdm(range(len(dataset['train']))):
                item = dataset['train'][i]
                example_url = "https://www.explainxkcd.com/wiki/images/5/5f/rnaworld_2x.png"
                test_url = item['image_url']

                # Fetch and process images
                try:
                    example_image = resize_transform(fetch_image(example_url))
                    test_image = resize_transform(fetch_image(test_url))

                    # Construct prompt for one example and one test image
                    example_prompt = (
                        "<|im_start|>user <image>\nAnalyze the comic and transcript and finally explain the humor (if any) along with the underlying message/concept it represents.|im_end|>"
                        "<|im_start|>assistant\n"
                        '''This comic humorously combines biology with Disney themes. It references the RNA world hypothesis, a theory on the origin of life where RNA serves both as genetic material and as a catalyst for its own replication.
Ariel (The Little Mermaid) is depicted collecting nucleotides instead of human artifacts, aligning with RNA and DNA building blocks.
Ratatouille, about a rat chef, is tied to "primordial soup," the early Earth's chemical environment where life began.
Elsa (Frozen), who brings a snowman to life with her ice powers, is linked to the emergence of life through ribozyme synthesis—RNA molecules that catalyze biochemical reactions.
The title text expands on this, suggesting Elsa's magic relies on ribozymes, making Olaf's life machinery RNA-based, consistent with the RNA World theme. It playfully notes Olaf, like cometary ice thought to deliver organic compounds to Earth, might be infused with carbon-rich molecules. The mention of "canonically" references official Disney lore.
'''
                        "|im_end|>"
                    )
                    test_prompt = (
                        "<|im_start|>user <image>\nAnalyze the comic and transcript and finally explain the humor (if any) along with the underlying message/concept it represents.|im_end|>"
                        "<|im_start|>assistant"
                    )
                    full_prompt = example_prompt + test_prompt

                    # Perform inference with the model
                    result = llava_inference_with_retry(full_prompt, [example_image, test_image])

                    if result:
                        json_line = {
                            "id": item['id'],
                            "example_image_url": example_url,
                            "test_image_url": test_url,
                            "response": result
                        }
                        f.write(json.dumps(json_line) + '\n')
                        f.flush()

                except Exception as e:
                    print(f"Error processing item {item['id']}: {e}")

if __name__ == "__main__":
    main()
