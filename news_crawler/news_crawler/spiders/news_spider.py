import scrapy
from scrapy import Request
from ..items import NewsItem
from bs4 import BeautifulSoup
import json
import random
import re
import jieba

TIME_PATTERN = '%Y-%m-%d %H:%M:%S'
SEARCH_PATTERN = 'https://so.news.cn/getNews?lang={lang}&curPage={page}\
&searchFields={only_title}&sortField={by_relativity}&keyword={keyword}'

# 默认参数
DEFAULT_LANGUAGE = 'cn'
DEFAULT_MAX_PAGES = 50
DEFAULT_NEWS_BATCH_SIZE = 100
DEFAULT_ONLY_TITLE = 0
DEFAULT_BY_RELATIVITY = 1

class NewsSpider(scrapy.Spider):
    name = 'news_spider'
    allowed_domains = ['news.cn', 'so.news.cn']
    
    def __init__(self, start_keyword='1', language=DEFAULT_LANGUAGE, max_pages=DEFAULT_MAX_PAGES,
                 news_batch_size=DEFAULT_NEWS_BATCH_SIZE, only_title=DEFAULT_ONLY_TITLE,
                 by_relativity=DEFAULT_BY_RELATIVITY, *args, **kwargs):
        super(NewsSpider, self).__init__(*args, **kwargs)
        
        # 初始化参数
        self.start_keyword = start_keyword
        self.language = language
        self.max_pages = int(max_pages)
        self.news_batch_size = int(news_batch_size)
        self.only_title = int(only_title)
        self.by_relativity = int(by_relativity)
        
        if self.language == 'cn':
            self.parse_news = self._parse_news_cn
            self.gen_keyword = self._gen_keyword_cn
        elif self.language == 'en':
            self.parse_news = self._parse_news_en
            self.gen_keyword = self._gen_keyword_en
        else:
            raise ValueError(f"Unsupported language: {self.language}")

        self.visited_urls = set()
        self.news_queue = []

    def start_requests(self):
        # 使用初始关键词 '1' 开始爬取
        keyword = self.start_keyword
        for page in range(1, self.max_pages+1):
            yield from self.search(page, keyword)
            
    def search(self, page, keyword):
        url = SEARCH_PATTERN.format(
            lang=self.language,
            page=page,
            only_title=self.only_title,
            by_relativity=self.by_relativity,
            keyword=keyword
        )
        yield Request(url, 
                       callback=self.parse_search, 
                       meta={'keyword': keyword, 'page': page},
                       priority=-page)

    def parse_search(self, response):
        keyword = response.meta['keyword']
        page = response.meta['page']
        self.logger.info(f"Searching for {keyword} Page {page}")
        item = None
        try:
            data = json.loads(response.text)
            news_list = data.get('content', {}).get('results', [])
            if not news_list:
                self.logger.warning(f"No news found for keyword '{keyword}' on page {page}.")
                return
            for news in news_list:
                url = news.get('url')
                if not url or url in self.visited_urls:
                    continue
                self.visited_urls.add(url)
                title = re.sub(r'<.*?>', '', news.get('title', ''))
                item = NewsItem()
                item['title'] = title.replace('&nbsp', ' ').replace(';', '').strip()
                item['time'] = news.get('pubtime')
                item['site'] = news.get('sitename')
                item['url'] = url
                self.news_queue.append(item)  # 将新闻加入队列
                
            # 如果队列大小超过一定数量，处理队列中的新闻
            if len(self.news_queue) >= self.news_batch_size:
                yield from self.process_news_queue()

                if item and item.get('title'):
                    title = item['title']
                    keyword = self.gen_keyword(title)
                else:
                    self.logger.warning("No item found to extract keyword from.")
                for page in range(1, self.max_pages+1):
                    yield from self.search(page, keyword)
                    
        except Exception as e:
            self.logger.error(f"Error parsing search response: {e}")

    def process_news_queue(self):
        while self.news_queue:
            news_item = self.news_queue.pop(0)
            yield Request(news_item['url'], 
                          callback=self.parse_news, 
                          meta={'item': news_item})
            
    def _parse_news_cn(self, response):
        item = response.meta['item']
        soup = BeautifulSoup(response.text, 'html.parser')
        if self.is_news(soup):
            detail = soup.find('div', id='detail')
            paragraphs = detail.find_all('p')
            item['content'] = '\n'.join([p.text.strip() for p in paragraphs])
            self.logger.info(f"Collected {item['title']}")
            yield item
        else:
            self.logger.warning(f"Not a news page: {item['url']}")
    
    def _parse_news_en(self, response):
        item = response.meta['item']
        soup = BeautifulSoup(response.text, 'html.parser')
        if self.is_news(soup):
            detail = soup.find('div', id='detail')
            paragraphs = detail.find_all('p')
            item['content'] = '\n'.join([p.text.strip() for p in paragraphs])
            self.logger.info(f"Collected {item['title']}")
            yield item
        else:
            self.logger.warning(f"Not a news page: {item['url']}")
            
    def _gen_keyword_cn(self, title: str):
        return random.choice(jieba.lcut(title))
    
    def _gen_keyword_en(self, title: str):
        return random.choice(title.split(' '))

    @staticmethod
    def is_news(soup):
        detail = soup.find('div', id='detail')
        return bool(detail)