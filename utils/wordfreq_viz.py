import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
import matplotlib.font_manager as fm

def plot_wordcloud(words: list[str], site: str):
    # 将词列表合并成一个字符串
    text = ' '.join(words)
    font_path = "./utils/SimHei.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    
    # 创建词云对象
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          max_words=200, 
                          colormap='viridis',
                          font_path=font_path).generate(text)
    
    # 绘制词云图
    plt.figure(figsize=(10, 5))
    plt.title(f'{site}词云图', fontproperties=font_prop, fontsize=16)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f'./docs/wordclouds/{site}.png')