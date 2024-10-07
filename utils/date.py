import re

def extract_date(text: str) -> list[str]:
    # 使用非捕获组 (?:) 避免捕获分隔符，如 "年", "月", "日"
    patterns = [
        r'\d{4}(?:年|-|/)\d{1,2}(?:月|-|/)\d{1,2}(?:日)?',  # 完整日期：YYYY-MM-DD
        r'\d{4}(?:年|-|/)\d{1,2}(?:月)?',                    # 年月：YYYY-MM
        r'\d{1,2}(?:月|-|/)\d{1,2}(?:日)?'                   # 月日：MM-DD
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates