import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from queue import Queue
import jieba
import random
import re

TIME_PATTERN = '%Y-%m-%d %H:%M:%S'
SEARCH_PATTERN = 'https://so.news.cn/getNews?lang={lang}&curPage={page}&\
searchFields={only_title}&sortField={by_relativity}&keyword={keyword}'
LANGUAGE = 'cn'
MAX_NEWS = 100
MAX_PAGES = 50

class News:
    def __init__(self, *, title=None, content=None, editor=None, site=None, time=None, url=None) -> None:
        self.title = title
        self.content = content
        self.editor = editor
        self.site = site
        self.time = time
        self.url = url

class NewsCrawler:
    def __init__(self) -> None:
        self.to_visit: Queue[News] = Queue()
        self.data: list[News] = []
        self.visited_urls: set[str] = set()
        
    def crawl(self):
        self.search('1')
        while True:
            news = self.to_visit.get()
            try:
                response = requests.get(news.url, timeout=3)
                soup = BeautifulSoup(response.text, 'html.parser')
                if self.is_news(soup):
                    news = self.get_news(soup, news)
                self.visited_urls.add(news.url)
                while self.to_visit.empty():
                    keyword = random.choice(jieba.lcut(news.title))
                    self.search(keyword)
                if len(self.data) >= MAX_NEWS:
                    break
            except TimeoutError as e:
                print(f"Timeout: {e} News: {news['title']}")
        
    def search(self, keyword: str):
        page = 1
        while page < MAX_PAGES:
            response = requests.get(SEARCH_PATTERN.format(lang=LANGUAGE, 
                                                          page=page,
                                                          only_title='title',
                                                          by_relativity='relativity',
                                                          keyword=keyword))
            print(f"Searching for {keyword} Page {page}")
            news_list = self.parse_search(response)
            if not news_list:
                break
            for news in news_list:
                self.to_visit.put(news)
            page += 1
            
    def parse_search(self, response: requests.Response) -> list[str]|None:
        news_list = []
        try:
            data = response.json()
            for news in data['content']['results']:
                title = re.sub(r'<.*?>', '', news['title'])
                news = News(title=title, 
                            time=news['pubtime'],
                            site=news['sitename'],
                            url=news['url'])
                if news.url in self.visited_urls:
                    continue
                news_list.append(news)
            return news_list
        except Exception as e:
            print(e)
            return None
        
    @staticmethod
    def is_news(soup: BeautifulSoup) -> bool:
        title = soup.find('span', class_='title')
        detail = soup.find('div', id='detail')
        if title and detail:
            return True
        else:
            return False
            
    def get_news(self, soup: BeautifulSoup, news: News) -> News:
        detail = soup.find('div', id='detail')
        paragraphs = detail.find_all('p')
        news.content = '\n'.join([p.text.strip() for p in paragraphs])
        editor = soup.find('span', class_='editor')
        news.editor = editor.text.strip() if editor else None
        self.data.append(news)
        print(f"Totoal: {len(self.data)} Collected {news.title}")
        return news
    
    def save_data(self, foldername: str) -> None:
        data_path = os.path.join(foldername, 'data.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump([news.__dict__ for news in self.data],
                      f, ensure_ascii=False, indent=4)
            
    def load_data(self, foldername: str) -> None:
        data_path = os.path.join(foldername, 'data.json')
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data = [News(**news) for news in data]
        except Exception as e:
            print(f"Failed to load data: {e}")