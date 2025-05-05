import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

SERPAPI_KEY = 'be10b3f377d4b728bbddb759fe7d1016850b859fa5517c4742d237e949991d86'  

def search_images_serpapi(query, max_results=100):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google",
        "tbm": "isch",
        "ijn": 0,
        "api_key": SERPAPI_KEY,
    }
    image_urls = []
    while len(image_urls) < max_results:
        params["ijn"] = len(image_urls) // 100
        response = requests.get(url, params=params)
        data = response.json()
        results = data.get("images_results", [])
        if not results:
            break
        for result in results:
            image_urls.append(result["original"])
            if len(image_urls) >= max_results:
                break
    return image_urls

def is_plain_background(pil_image, edge_thresh=0.07):
    img = pil_image.resize((100, 100))
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    return edge_ratio < edge_thresh

def download_and_filter_images(image_urls, save_dir, label, max_save=300):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for i, url in enumerate(image_urls):
        try:
            img = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
            if is_plain_background(img):
                img.save(os.path.join(save_dir, f"{label}_{count:04d}.jpg"))
                count += 1
                print(f"[{label}] Saved {count}/{max_save}", end='\r')
            if count >= max_save:
                break
        except:
            continue
    print(f"\n{label} - Done. {count} plain-background images saved.")

keywords = ["plain background knife", "plain background grenade", "plain background handgun"]
base_dir = "./images"

for keyword in keywords:
    label = keyword.split()[-1]
    print(f"\n🔍 Searching for: {keyword}")
    urls = search_images_serpapi(keyword + " on white background", max_results=400)
    download_and_filter_images(urls, os.path.join(base_dir, label), label, max_save=100)
