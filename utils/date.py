import re

def extract_date(text: str) -> list[str]:
    # 定义正则模式，按照从具体到模糊的顺序排列
    patterns = [
        # 完整日期：YYYY年MM月DD日 或 YYYY-MM-DD 或 YYYY/MM/DD
        r'\b\d{4}(?:年|-|/)(?:0?[1-9]|1[0-2])(?:月|-|/)(?:0?[1-9]|[12]\d|3[01])(?:日)?\b',
        # 年月：YYYY年MM月 或 YYYY-MM 或 YYYY/MM
        r'\b\d{4}(?:年|-|/)(?:0?[1-9]|1[0-2])(?:月)?\b',
        # 月日：MM月DD日 或 MM-DD 或 MM/DD
        r'\b(?:0?[1-9]|1[0-2])(?:月|-|/)(?:0?[1-9]|[12]\d|3[01])(?:日)?\b'
    ]
    
    dates = []
    for pattern in patterns:
        # 使用 re.findall 找到所有匹配，并添加到 dates 列表
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    # 去重，同时保持顺序
    seen = set()
    unique_dates = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)
    
    return unique_dates