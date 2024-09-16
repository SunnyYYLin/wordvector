from crawler.news import NewsCrawler

if __name__ == '__main__':
    crawler = NewsCrawler('http://www.news.cn')
    crawler.get_urls()
    for url in crawler.urls:
        crawler.get_news(url)
    crawler.save_data('data.json')