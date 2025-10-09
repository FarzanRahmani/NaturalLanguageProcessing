import ssl
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import os 
import warnings
import urllib3
import pandas as pd
from tqdm import tqdm

class CustomHttpAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.set_ciphers('DEFAULT:@SECLEVEL=1')  # Allow less secure ciphers
        kwargs['ssl_context'] = ctx
        return super(CustomHttpAdapter, self).init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', CustomHttpAdapter())

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome()

def extract_infobox_with_images(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    infobox = soup.find('table', class_='infobox')
    if not infobox:
        return "No infobox found."

    info_dict = {}

    # Try to extract the title/name
    name_tag = infobox.find(['div', 'caption'], class_='fn')
    if name_tag:
        info_dict['Name'] = name_tag.get_text(strip=True)

    # Traverse rows
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

    # Collect image URLs and their captions
    images_info = []
    for img_td in infobox.find_all('td', class_='infobox-image'):
        img_tag = img_td.find('img')
        if img_tag:
            src = img_tag.get('src')
            if src and src.startswith('//'):
                src = 'https:' + src
            caption_tag = img_td.find_next('div', class_='infobox-caption')
            caption = caption_tag.get_text(strip=True) if caption_tag else 'No caption'
            images_info.append((src, caption))

    # Construct final output
    result = ''
    for key, value in info_dict.items():
        result += f"{key}: {value} - "

    if images_info:
        result += "\nImages:\n"
        for idx, (src, caption) in enumerate(images_info, start=1):
            result += f"{idx}. {caption}\n   URL: {src}\n"

    return result

def extract_llm_ready_text(html_content):
    # 1. Cut everything from "See also" section onwards
    stop_marker = '<div class="mw-heading mw-heading2"><h2 id="See_also">See also</h2>'
    cut_index = html_content.find(stop_marker)
    if cut_index != -1:
        html_content = html_content[:cut_index]

    # 2. Remove navboxes
    soup = BeautifulSoup(html_content, "html.parser")
    for navbox in soup.find_all("div", class_="navbox"):
        navbox.decompose()

    # 3. Paragraphs
    p_texts = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
    
    # 4. Lists (ul + dl pairing)
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

    # 5. Figures / Images / Math
    images_info = set()

    for fig in soup.find_all('figure'):
        img = fig.find('img')
        caption = fig.find('figcaption')
        if img:
            src = img.get('src')
            if src and src.startswith('//'):
                src = 'https:' + src
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

    # 6. Blockquotes
    blockquotes = [bq.get_text(separator=' ', strip=True) for bq in soup.find_all('blockquote')]

    # 7. Combine all extracted text
    combined_text = (
        "\n".join(p_texts) +
        "\n".join(ul_texts) +
        "\n".join(images_info) +
        "\n".join(blockquotes)
    )

    return combined_text.strip()


link_df_en = pd.read_csv("ir_treaty_cleaned.csv")

# Output file path
output_csv_path = "ir_treaty_cleaned_en_output_v2.csv"

# Check if the file exists to decide whether to write headers
file_exists = os.path.exists(output_csv_path)

for index, row in tqdm(link_df_en.iterrows()):
    name = row['names']
    link = row['link']

    try:
        driver.get(link)
        time.sleep(1)
    except Exception:
        print(f"Something went wrong accessing: {link}")
        continue

    try:
        content = driver.find_element(By.CLASS_NAME, "mw-content-ltr")
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

    # Combine and prepare the row
    merged_text = infobox.strip() + clean_text_for_llm.strip()
    new_row = pd.DataFrame([{
        "name": name,
        "link": link,
        "content": merged_text
    }])

    # Append to CSV
    new_row.to_csv(output_csv_path, mode='a', header=not file_exists, index=False, encoding='utf-8')
    file_exists = True  # After first write, disable header writing

# print(pd.read_csv(output_csv_path).shape)
    

