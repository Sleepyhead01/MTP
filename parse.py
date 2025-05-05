# import argparse

# from vllm import LLM
# from vllm.sampling_params import SamplingParams

# # This script is an offline demo for running Pixtral.
# #
# # If you want to run a server/client setup, please follow this code:
# #
# # - Server:
# #
# # ```bash
# # vllm serve mistralai/Pixtral-12B-2409 --tokenizer-mode mistral --limit-mm-per-prompt 'image=4' --max-model-len 16384
# # ```
# #
# # - Client:
# #
# # ```bash
# # curl --location 'http://<your-node-url>:8000/v1/chat/completions' \
# # --header 'Content-Type: application/json' \
# # --header 'Authorization: Bearer token' \
# # --data '{
# #     "model": "mistralai/Pixtral-12B-2409",
# #     "messages": [
# #       {
# #         "role": "user",
# #         "content": [
# #             {"type" : "text", "text": "Describe this image in detail please."},
# #             {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}},
# #             {"type" : "text", "text": "and this one as well. Answer in French."},
# #             {"type": "image_url", "image_url": {"url": "https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png"}}
# #         ]
# #       }
# #     ]
# #   }'
# # ```
# #
# # Usage:
# #     python demo.py simple
# #     python demo.py advanced


# from vllm import LLM
# from vllm.sampling_params import SamplingParams

# def run_advanced_demo():
#     model_name = "mistralai/Pixtral-12B-2409"
#     max_img_per_msg = 5
#     max_tokens_per_img = 4096

#     sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)
#     llm = LLM(
#         model=model_name,
#         tokenizer_mode="mistral",
#         limit_mm_per_prompt={"image": max_img_per_msg},
#         max_model_len=max_img_per_msg * max_tokens_per_img,
#     )

#     prompt = "Describe the following image."

#     url_1 = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
#     url_2 = "https://picsum.photos/seed/picsum/200/300"
#     url_3 = "https://picsum.photos/id/32/512/512"

#     messages = [
#         {
#             "role":
#             "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": prompt
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": url_1
#                     }
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": url_2
#                     }
#                 },
#             ],
#         },
#         {
#             "role": "assistant",
#             "content": "The images show nature.",
#         },
#         {
#             "role": "user",
#             "content": "More details please and answer only in French!.",
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": url_3
#                     }
#                 },
#             ],
#         },
#     ]

#     outputs = llm.chat(messages=messages, sampling_params=sampling_params)
#     print(outputs[0].outputs[0].text)


# def main():
#     print("Running advanced demo...")
#     run_advanced_demo()


# if __name__ == "__main__":
#     main()









# !pip install selenium bs4 undetected_chromedriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
import undetected_chromedriver as uc
import requests
from bs4 import BeautifulSoup
import pandas as pd
# import time
from time import sleep
from bs4 import BeautifulSoup


options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument("--headless")
options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")
# options.add_argument("--remote-debugging-port=9222")
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option('useAutomationExtension', False)
# user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
# options.add_argument(f'user-agent={user_agent}')
# driver = uc.Chrome(options=options)
driver = webdriver.Chrome(options=options)

def go_to(locator:tuple, driver=driver, duration:int=60):
    element = WebDriverWait(driver, duration).until(
        EC.element_to_be_clickable(locator)
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    sleep(1)
    return element 

RANGE = '2501-3000'
driver.get(f'https://www.explainxkcd.com/wiki/index.php/List_of_all_comics_({RANGE})')

page_source = driver.page_source
# print(page_source)

import json
from bs4 import BeautifulSoup

# Assuming the driver and page source are already initialized
soup = BeautifulSoup(page_source, 'html.parser')
table = soup.find('table', class_='wikitable sortable plainlinks table-padding jquery-tablesorter')

# Step 3: Extract table headers
headers = [th.text.strip() for th in table.find_all('th')]
headers.append('urls')

# Step 4: Extract rows and data
rows = table.find_all('tr')
data = []
links = []
for row in rows[1:]:  # Skip the header row
    cells = row.find_all(['td', 'th'])
    data.append([cell.text.strip() for cell in cells])
    links.append(cells[1].find('a')['href'])

data = [data[i] + [links[i]] for i in range(len(data))]

# Save extracted table data in JSON Lines (optional)
with open('table_data.jsonl', 'w') as f:
    for row in data:
        json.dump(dict(zip(headers, row)), f)
        f.write('\n')

# Step 5: Iterate through URLs and process data
with open('xkcd_comics.jsonl', 'w') as f:
    for i in range(len(data)):
        # print("Processing idx: ", i)
        driver.get('https://www.explainxkcd.com' + data[i][-1])  # Access 'urls' column
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # print("Processing comic number", soup.find('h1', class_='firstHeading').text.strip())
        text = soup.find('div', id='bodyContent').text

        # Find the img URL
        table = soup.find('table', class_='comic-content')
        img_tag = table.find('img')['src']

        # Prepare comic data
        comic_data = {
            'title': soup.find('h1', class_='firstHeading').text.split(':')[1].strip(),
            'explanation': text.split('Explanation[edit]')[1].split('Transcript[edit]')[0].strip(),
            'transcript': text.split('Transcript[edit]')[1].split('add a comment!')[0].strip(),
            'img_url': 'https://www.explainxkcd.com' + img_tag
        }

        # Save comic data to JSON Lines file
        json.dump(comic_data, f)
        f.write('\n')
        print(f"Saved comic {comic_data['title']} to file.")
