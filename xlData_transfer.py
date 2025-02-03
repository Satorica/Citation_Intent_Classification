# -*- coding: utf-8 -*-
import pandas as pd
import re
import json

# 加载CSV文件
df = pd.read_csv('./xlData/ARC_all_labeled.csv')

# 定义一个函数来替换引用
def replace_citation(text, citing_author_year):
    # 提取作者和年份
    authors_year = citing_author_year.split(',')
    authors = authors_year[:-1]
    year = authors_year[-1].strip()
    
    # 为每个作者构建正则表达式
    patterns = []
    for author in authors:
        # 提取姓氏
        surname = author.split()[-1]
        # 构建匹配姓氏部分字符的正则表达式
        pattern = re.escape(surname[0].upper()) + re.escape(surname[1:]) + r'[A-Za-z\(\s,.]*?' + re.escape(year)
        patterns.append(pattern)
    
    # 组合多个作者的正则表达式
    # 匹配任意一个作者的模式，并确保年份存在
    combined_pattern = r'(?:' + r'|'.join(patterns) + r')'
    print(combined_pattern)

    match = re.search(combined_pattern, text)

    if match:
        # 替换为@@CITATION
        replaced_text = re.sub(combined_pattern, '@@CITATION', text, count=1)
        return replaced_text, True
    else: 
        return text, False

# 应用替换函数到每一行
# df['CitationContent'] = df.apply(lambda row: replace_citation(row['CitationContent'], row['CitedAuthorYear']), axis=1)

# 构建JSON结构
# 打开文件，准备逐行写入
with open('output.jsonl', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        replaced_text, is_matched = replace_citation(row['CitationContent'], row['CitedAuthorYear'])
        if is_matched:
            result = {
                "text": row['CitationContent'],
                "citing_paper_id": row['CitingpaperID'],
                "cited_paper_id": row['CitedpaperID'],
                "citing_paper_year": int(row['CitingAuthorYear'].split(',')[-1].strip()),
                "cited_paper_year": int(row['CitedAuthorYear'].split(',')[-1].strip()),
                "citing_paper_title": row['CitingTitleAbstract'],
                "cited_paper_title": row['CitedTitleAbstract'],
                "cited_author_ids": row['CitedAuthorYear'].split(',')[:-1],
                "citing_author_ids": row['CitingAuthorYear'].split(',')[:-1],
                "extended_context": replaced_text,
                "section_number": None,
                "section_title": None,
                "intent": row['LabelType'],
                "cite_marker_offset": [0, 0],
                "cleaned_cite_text": replaced_text,
                "citation_id": f"{row['CitingpaperID']}_{_}",
                "citation_excerpt_index": 0,
                "section_name": "introduction"
            }
            # 逐行写入文件（每行一个独立的JSON对象）
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

print("save 'output.json'")