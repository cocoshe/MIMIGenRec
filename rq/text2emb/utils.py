import collections
import html
import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import requests
import torch

# import gensim
from transformers import AutoModel, AutoTokenizer


def get_res_batch(model_name, prompt_list, max_tokens, api_info):
    provider = api_info.get("provider", "openai")

    if provider == "deepseek":
        return get_deepseek_batch(model_name, prompt_list, max_tokens, api_info)
    else:
        return get_openai_batch(model_name, prompt_list, max_tokens, api_info)


def get_openai_batch(model_name, prompt_list, max_tokens, api_info):
    while True:
        try:
            res = openai.Completion.create(
                model=model_name,
                prompt=prompt_list,
                temperature=0.4,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            output_list = []
            for choice in res["choices"]:
                output = choice["text"].strip()
                output_list.append(output)

            return output_list

        except openai.error.AuthenticationError as e:
            print(e)
            openai.api_key = api_info["api_key_list"].pop()
            time.sleep(10)
        except openai.error.RateLimitError as e:
            print(e)
            if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                openai.api_key = api_info["api_key_list"].pop()
                time.sleep(10)
            else:
                print("\nopenai.error.RateLimitError\nRetrying...")
                time.sleep(10)
        except openai.error.ServiceUnavailableError as e:
            print(e)
            print("\nopenai.error.ServiceUnavailableError\nRetrying...")
            time.sleep(10)
        except openai.error.Timeout:
            print("\nopenai.error.Timeout\nRetrying...")
            time.sleep(10)
        except openai.error.APIError as e:
            print(e)
            print("\nopenai.error.APIError\nRetrying...")
            time.sleep(10)
        except openai.error.APIConnectionError as e:
            print(e)
            print("\nopenai.error.APIConnectionError\nRetrying...")
            time.sleep(10)
        except Exception as e:
            print(e)
            return None


def get_deepseek_batch(model_name, prompt_list, max_tokens, api_info):
    base_url = api_info.get("base_url", "https://api.deepseek.com")

    max_workers = min(256, len(prompt_list))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, prompt in enumerate(prompt_list):
            future = executor.submit(_single_deepseek_request, model_name, prompt, max_tokens, api_info, base_url, i)
            futures.append(future)

        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                print(f"请求 {i + 1} 失败: {e}")
                results.append("")

        return results


def _single_deepseek_request(model_name, prompt, max_tokens, api_info, base_url, request_id):
    api_key = api_info["api_key_list"][-1]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
    }

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # 添加随机延时避免同时请求
            import random

            delay = random.uniform(0.01, 0.03) * request_id
            time.sleep(delay)

            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                return content

            elif response.status_code == 429:
                retry_delay = (2**retry_count) * random.uniform(1, 2)
                print(f"请求 {request_id + 1} 遇到速率限制，等待 {retry_delay:.1f}s 后重试")
                time.sleep(retry_delay)
                retry_count += 1
                continue

            else:
                print(f"请求 {request_id + 1} 错误: {response.status_code}")
                retry_count += 1
                time.sleep(1)
                continue

        except Exception as e:
            print(f"请求 {request_id + 1} 异常: {e}")
            retry_count += 1
            time.sleep(1)

    return ""


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")


def load_plm(model_path="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )

    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def write_json_file(dic, file):
    print("Writing json file: ", file)
    with open(file, "w") as fp:
        json.dump(dic, fp, indent=4)


def write_remap_index(unit2index, file):
    print("Writing remap file: ", file)
    with open(file, "w") as fp:
        for unit in unit2index:
            fp.write(unit + "\t" + str(unit2index[unit]) + "\n")


intention_prompt = (
    'After purchasing a {dataset_full_name} item named "{item_title}", the user left a comment expressing his opinion and personal preferences. The user\'s comment is as follows: \n"{review}" '
    "\nAs we all know, user comments often contain information about both their personal preferences and the characteristics of the item they interacted with. From this comment, you can infer both the user's personal preferences and the characteristics of the item. "
    "Please describe your inferred user preferences and item characteristics in the first person and in the following format:\n\nMy preferences: []\nThe item's characteristics: []\n\n"
    "Note that your inference of the personalized preferences should not include any information about the title of the item."
)


preference_prompt_1 = (
    "A user has bought a variety of {dataset_full_name} items in chronological order: \n{item_titles}. \nAfter purchasing these items, the user then chose to buy the following item: {target_item_info}. "
    "Based on this purchasing pattern and the final choice, please analyze what this reveals about the user's personalized preferences. Provide a brief third-person summary that explains the logical progression from the historical purchases to this final choice, highlighting the key factors that influence the user's decisions. "
    "In your analysis, do not mention the specific name/title of the target item, but focus on the characteristics and features that drove the user to make this particular choice. Your analysis should be brief and in the third person."
)

preference_prompt_2 = (
    "A user has purchased a chronological list of {dataset_full_name} items: \n{item_titles}. \nThen, the user made a final purchase: {target_item_info}. "
    "Based on this purchasing progression, please provide a concise analysis of what drove the user to choose this specific type of item. Focus on the logical connection between the user's purchase history and this final choice. "
    "Explain the purchasing pattern and what it reveals about the user's evolving needs or interests. Your analysis should be brief and in the third person."
)


# remove 'Magazine', 'Gift', 'Music', 'Kindle'
amazon18_dataset_list = [
    "Appliances",
    "Beauty",
    "Fashion",
    "Software",
    "Luxury",
    "Scientific",
    "Pantry",
    "Instruments",
    "Arts",
    "Games",
    "Office",
    "Garden",
    "Food",
    "Cell",
    "CDs",
    "Automotive",
    "Toys",
    "Pet",
    "Tools",
    "Kindle",
    "Sports",
    "Movies",
    "Electronics",
    "Home",
    "Clothing",
    "Books",
]

amazon18_dataset2fullname = {
    "Beauty": "All_Beauty",
    "Fashion": "AMAZON_FASHION",
    "Appliances": "Appliances",
    "Arts": "Arts_Crafts_and_Sewing",
    "Automotive": "Automotive",
    "Books": "Books",
    "CDs": "CDs_and_Vinyl",
    "Cell": "Cell_Phones_and_Accessories",
    "Clothing": "Clothing_Shoes_and_Jewelry",
    "Music": "Digital_Music",
    "Electronics": "Electronics",
    "Gift": "Gift_Cards",
    "Food": "Grocery_and_Gourmet_Food",
    "Home": "Home_and_Kitchen",
    "Industrial_and_Scientific": "Industrial_and_Scientific",
    "Kindle": "Kindle_Store",
    "Luxury": "Luxury_Beauty",
    "Magazine": "Magazine_Subscriptions",
    "Movies": "Movies_and_TV",
    "Instruments": "Musical_Instruments",
    "Office": "Office_Products",
    "Garden": "Patio_Lawn_and_Garden",
    "Pet": "Pet_Supplies",
    "Pantry": "Prime_Pantry",
    "Software": "Software",
    "Sports": "Sports_and_Outdoors",
    "Tools": "Tools_and_Home_Improvement",
    "Toys_and_Games": "Toys_and_Games",
    "Games": "Video_Games",
}

amazon14_dataset_list = ["Beauty", "Toys", "Sports"]

amazon14_dataset2fullname = {
    "Beauty": "Beauty",
    "Sports": "Sports_and_Outdoors",
    "Toys": "Toys_and_Games",
}

# c1. c2. c3. c4.
amazon_text_feature1 = ["title", "category", "brand"]

# re-order
amazon_text_feature1_ro1 = ["brand", "main_cat", "category", "title"]

# remove
amazon_text_feature1_re1 = ["title"]

amazon_text_feature2 = ["title"]

amazon_text_feature3 = ["description"]

amazon_text_feature4 = ["description", "main_cat", "category", "brand"]

amazon_text_feature5 = ["title", "description"]
