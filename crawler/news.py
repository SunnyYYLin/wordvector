import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

TIME_PATTERN = '%Y/%m/%d %H:%M:%S'
MAX_NEWS = 100

class News:
    def __init__(self, title, info, content, editor, topic, time) -> None:
        self.title = title
        self.info = info
        self.content = content
        self.editor = editor
        self.topic = topic
        self.time = time

class NewsCrawler:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.urls: set[str] = set()
        self.data: list[News] = []
        
    @staticmethod
    def is_news(soup: BeautifulSoup) -> bool:
        title = soup.find('span', class_='title')
        detail = soup.find('div', id='detail')
        if title and detail:
            return True
        else:
            return False
        
    @staticmethod
    def parse_time(time_soup: BeautifulSoup) -> datetime:
        year = time_soup.find('span', class_='year').text
        day = time_soup.find('span', class_='day').text
        time = time_soup.find('span', class_='time').text
        return datetime.strptime(f'{year}/{day} {time}', TIME_PATTERN)
        
    def get_urls(self) -> set[str]:
        urls_to_visit = set([self.base_url])
        urls_visited = set()
        while True:
            url = urls_to_visit.pop()
            urls_visited.add(url)
            try:
                response = requests.get(url)
            except:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            if self.is_news(soup):
                self.urls.add(url)
            for a in soup.find_all('a', href=True):
                href: str = a['href']
                if not href.startswith(self.base_url):
                    href = self.base_url + href
                if href not in urls_visited:
                    urls_to_visit.add(href)
            if not urls_to_visit or len(self.urls) >= MAX_NEWS:
                break
        return self.urls
            
    def get_news(self, url: str) -> News:
        response = requests.get(url)
        topic = url.split('/')[3]
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('span', class_='title').text.strip()
        info = soup.find('span', class_='info')
        info = info.text.strip() if info else ''
        detail = soup.find('div', id='detail')
        paragraphs = detail.find_all('p')
        content = '\n'.join([p.text.strip() for p in paragraphs])
        editor = soup.find('span', class_='editor').text.strip()
        time = self.parse_time(soup.find('div', class_='header-time left'))
        news = News(title, info, content, editor, topic, datetime.strftime(time, TIME_PATTERN))
        self.data.append(news)
        return news
    
    def save_data(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([news.__dict__ for news in self.data],
                      f, ensure_ascii=False, indent=4)
            
    def load_data(self, filename: str) -> None:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.data = [News(**news) for news in data]