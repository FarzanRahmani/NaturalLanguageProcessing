import ssl
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import os 
import warnings
import urllib3
import pandas as pd
import csv
from tqdm import tqdm

class CustomHttpAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.set_ciphers('DEFAULT:@SECLEVEL=1')  # Allow less secure ciphers
        kwargs['ssl_context'] = ctx
        return super(CustomHttpAdapter, self).init_poolmanager(*args, **kwargs)

# Setup session and driver
session = requests.Session()
session.mount('https://', CustomHttpAdapter())

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

def extract_infobox_with_images(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    infobox = soup.find('table', class_='infobox')
    if not infobox:
        return ""

    info_dict = {}

    # Name (from <div class="fn"> or fallback)
    name_tag = infobox.find(['div', 'caption'], class_='fn')
    if name_tag:
        info_dict['Name'] = name_tag.get_text(strip=True)

    # Key-value pairs
    for row in infobox.find_all('tr'):
        header = row.find('th')
        data = row.find('td')
        if header and data:
            key = header.get_text(strip=True)
            if data.find('ul'):
                values = [li.get_text(strip=True) for li in data.find_all('li')]
                info_dict[key] = ', '.join(values)
            else:
                info_dict[key] = data.get_text(separator=' ', strip=True)

    # Images + captions
    images = []
    for img_td in infobox.find_all('td', class_='infobox-image'):
        img_tag = img_td.find('img')
        if img_tag:
            src = img_tag.get('src')
            if src and src.startswith('//'):
                src = 'https:' + src
            caption_tag = img_td.find_next('div', class_='infobox-caption')
            caption = caption_tag.get_text(strip=True) if caption_tag else 'No caption'
            images.append(f"{caption}: {src}")

    # Assemble full infobox string
    full_string = ''
    for key, value in info_dict.items():
        full_string += f"{key}: {value}\n"
    for img in images:
        full_string += f"{img}\n"

    return full_string.strip()

def extract_llm_ready_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted sections
    # Remove navboxes
    for tag in soup.select(".navbox, .navbox-styles"):
        tag.decompose()

    # Find the "جستارهای وابسته" heading and stop parsing after it
    stop_heading = soup.find("h2", id="جستارهای_وابسته")
    if stop_heading:
        for elem in list(stop_heading.find_all_next()):
            elem.decompose()

    # Paragraphs
    p_texts = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]

    # Lists (ul + optional preceding dl)
    ul_texts = []
    for ul in soup.find_all('ul'):
        dl = ul.find_previous_sibling('dl')
        dl_text = dl.get_text(separator=' ', strip=True) if dl else None
        items = [li.get_text(strip=True) for li in ul.find_all('li')]
        ul_line = ", ".join(items)
        if dl_text:
            ul_texts.append(f"{dl_text}: {ul_line}")
        else:
            ul_texts.append(ul_line)

    # Images and Figures
    images_info = set()
    for fig in soup.find_all('figure'):
        img = fig.find('img')
        caption = fig.find('figcaption')
        if img:
            src = img.get("src")
            if src and src.startswith("//"):
                src = "https:" + src
            caption_text = caption.get_text(strip=True) if caption else ""
            if caption_text and src:
                images_info.add(f"{caption_text}: {src}")

    for span in soup.find_all("span", class_="mwe-math-element"):
        img = span.find("img")
        if img:
            alt = img.get("alt", "").strip()
            src = img.get("src")
            if src and src.startswith("//"):
                src = "https:" + src
            if alt and src:
                images_info.add(f"{alt}: {src}")

    for img in soup.find_all("img"):
        src = img.get("src")
        if src and src.startswith("//"):
            src = "https:" + src
        alt_text = img.get("alt", "").strip()
        caption = img.find_next("div", class_="thumbcaption")
        caption_text = caption.get_text(strip=True) if caption else alt_text
        if src and caption_text:
            images_info.add(f"{caption_text}: {src}")

    # Blockquotes
    blockquotes = [bq.get_text(separator=' ', strip=True) for bq in soup.find_all('blockquote')]

    # Combine all into one plain text string
    all_text = (
        "\n".join(p_texts) +
        "\n".join(ul_texts) +
        "\n".join(images_info) +
        "\n".join(blockquotes)
    )

    return all_text.strip()


# Setup headless Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

# Input/Output
link_df_fa = pd.read_csv("wars_links_cleaned.csv")
output_csv_path = "ewars_links_cleaned_fa_output_V2.csv"
file_exists = os.path.exists(output_csv_path)

# Main loop
for index, row in tqdm(link_df_fa.iterrows()):
    name = row['names']
    link = row['link']

    try:
        driver.get(link)
        time.sleep(1)
    except Exception:
        print(f"Something went wrong accessing: {link}")
        continue

    try:
        content = driver.find_element(By.CLASS_NAME, "mw-parser-output")  # Use RTL-compatible class
        html_string = content.get_attribute("outerHTML")
    except Exception:
        print(f"{link} - could not extract content element")
        continue

    try:
        infobox = extract_infobox_with_images(html_string)
    except Exception:
        print(f"{link} - problem in infobox extraction")
        infobox = ""

    try:
        clean_text_for_llm = extract_llm_ready_text(html_string)
    except Exception:
        print(f"{link} - problem in body extraction")
        clean_text_for_llm = ""

    merged_text = infobox.strip() + "\n" + clean_text_for_llm.strip()

    new_row = pd.DataFrame([{
        "name": name,
        "link": link,
        "content": merged_text
    }])

    new_row.to_csv(output_csv_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
    file_exists = True  # Set true after writing header once
    
    
# print(pd.read_csv(output_csv_path).shape)

