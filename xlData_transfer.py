# -*- coding: utf-8 -*-
import pandas as pd
import re
import json

# ����CSV�ļ�
df = pd.read_csv('./xlData/ARC_all_labeled.csv')

# ����һ���������滻����
def replace_citation(text, citing_author_year):
    # ��ȡ���ߺ����
    authors_year = citing_author_year.split(',')
    authors = authors_year[:-1]
    year = authors_year[-1].strip()
    
    # Ϊÿ�����߹���������ʽ
    patterns = []
    for author in authors:
        # ��ȡ����
        surname = author.split()[-1]
        # ����ƥ�����ϲ����ַ���������ʽ
        pattern = re.escape(surname[0].upper()) + re.escape(surname[1:]) + r'[A-Za-z\(\s,.]*?' + re.escape(year)
        patterns.append(pattern)
    
    # ��϶�����ߵ�������ʽ
    # ƥ������һ�����ߵ�ģʽ����ȷ����ݴ���
    combined_pattern = r'(?:' + r'|'.join(patterns) + r')'
    print(combined_pattern)

    match = re.search(combined_pattern, text)

    if match:
        # �滻Ϊ@@CITATION
        replaced_text = re.sub(combined_pattern, '@@CITATION', text, count=1)
        return replaced_text, True
    else: 
        return text, False

# Ӧ���滻������ÿһ��
# df['CitationContent'] = df.apply(lambda row: replace_citation(row['CitationContent'], row['CitedAuthorYear']), axis=1)

# ����JSON�ṹ
# ���ļ���׼������д��
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
            # ����д���ļ���ÿ��һ��������JSON����
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

print("save 'output.json'")