import os
import requests
import urllib.parse
import cv2
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
import io

# 配置参数
SEARCH_KEYWORD = "张维维"  # 替换目标名人
DOWNLOAD_COUNT = 20
SAVE_PATH = os.path.abspath("celebrity_images")
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://cn.bing.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9'
}

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def fetch_image_urls(keyword, max_count):
    """获取经过验证的图片URL"""
    encoded_keyword = urllib.parse.quote(keyword)
    image_urls = []
    
    try:
        search_url = f"https://cn.bing.com/images/search?q={encoded_keyword}&qft=+filterui:imagesize-large"
        print(f"正在访问搜索页面: {search_url}")
        
        response = requests.get(search_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all('img', class_='mimg')
        
        print(f"发现{len(img_tags)}个图片标签")
        
        for img in img_tags:
            if len(image_urls) >= max_count:
                break
            img_url = img.get('src') or img.get('data-src')
            if img_url:
                img_url = urllib.parse.unquote(img_url)
                if img_url.startswith('http'):
                    image_urls.append(img_url)
                    print(f"有效URL: {img_url}")
                else:
                    print(f"忽略无效URL: {img_url}")
        
    except Exception as e:
        print(f"搜索页面请求失败: {str(e)}")
    
    return image_urls[:max_count]

def download_and_filter_images(urls):
    """增强的下载与过滤逻辑"""
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    success_count = 0
    for idx, url in enumerate(urls):
        try:
            print(f"\n正在处理 [{idx+1}/{len(urls)}]: {url}")
            
            # 下载并验证图片
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            
            # 内存验证图片格式
            try:
                img = Image.open(io.BytesIO(response.content))
                img.verify()
            except Exception as e:
                print(f"图片验证失败: {str(e)}")
                continue
            
            # 人脸检测
            img_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            if img_array is None:
                print("无法解码图片")
                continue
                
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                save_path = os.path.join(SAVE_PATH, f"{SEARCH_KEYWORD}_{success_count+1}.jpg")
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
                print(f"√ 成功保存: {save_path}")
            else:
                print("未检测到有效人脸")
                
        except Exception as e:
            print(f"处理失败: {str(e)}")
            continue
            
    print(f"\n完成！成功下载 {success_count}/{len(urls)} 张图片")

if __name__ == "__main__":
    print(f"保存路径: {SAVE_PATH}")
    image_urls = fetch_image_urls(SEARCH_KEYWORD, DOWNLOAD_COUNT)
    if image_urls:
        download_and_filter_images(image_urls)
    else:
        print("错误：未能获取任何图片URL")