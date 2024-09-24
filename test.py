import json
import requests
from bs4 import BeautifulSoup

def is_news(soup):
    detail = soup.find('div', id='detail')
    return bool(detail)

res = requests.get('https://english.news.cn/20240106/769e46e0c3f14d20ab73f94fae04ca6c/c.html')
print(res.text)
sp = BeautifulSoup(res.text, 'html.parser')
print(is_news(sp))