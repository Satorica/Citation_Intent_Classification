# # -*- coding: utf-8 -*-
# import pandas as pd
# import re
# import json

# # 加载CSV文件
# df = pd.read_csv('./xlData/ARC_all_labeled.csv')

# # 定义一个函数来替换引用
# def replace_citation(text, citing_author_year):
#     # 提取作者和年份
#     authors_year = citing_author_year.split(',')
#     authors = authors_year[:-1]
#     year = authors_year[-1].strip()
    
#     # 为每个作者构建正则表达式
#     patterns = []
#     for author in authors:
#         # 提取姓氏
#         surname = author.split()[-1]
#         # 构建匹配姓氏部分字符的正则表达式
#         pattern = re.escape(surname[0].upper()) + re.escape(surname[1:]) + r'[A-Za-z\(\s,.]*?' + re.escape(year)
#         patterns.append(pattern)
    
#     # 组合多个作者的正则表达式
#     # 匹配任意一个作者的模式，并确保年份存在
#     combined_pattern = r'(?:' + r'|'.join(patterns) + r')'
#     print(combined_pattern)

#     match = re.search(combined_pattern, text)

#     if match:
#         # 替换为@@CITATION
#         replaced_text = re.sub(combined_pattern, '@@CITATION', text, count=1)
#         return replaced_text, True
#     else: 
#         return text, False

# # 应用替换函数到每一行
# # df['CitationContent'] = df.apply(lambda row: replace_citation(row['CitationContent'], row['CitedAuthorYear']), axis=1)

# # 构建JSON结构
# # 打开文件，准备逐行写入
# with open('output.jsonl', 'w', encoding='utf-8') as f:
#     for _, row in df.iterrows():
#         replaced_text, is_matched = replace_citation(row['CitationContent'], row['CitedAuthorYear'])
#         if is_matched:
#             result = {
#                 "text": row['CitationContent'],
#                 "citing_paper_id": row['CitingpaperID'],
#                 "cited_paper_id": row['CitedpaperID'],
#                 "citing_paper_year": int(row['CitingAuthorYear'].split(',')[-1].strip()),
#                 "cited_paper_year": int(row['CitedAuthorYear'].split(',')[-1].strip()),
#                 "citing_paper_title": row['CitingTitleAbstract'],
#                 "cited_paper_title": row['CitedTitleAbstract'],
#                 "cited_author_ids": row['CitedAuthorYear'].split(',')[:-1],
#                 "citing_author_ids": row['CitingAuthorYear'].split(',')[:-1],
#                 "extended_context": replaced_text,
#                 "section_number": None,
#                 "section_title": None,
#                 "intent": row['LabelType'],
#                 "cite_marker_offset": [0, 0],
#                 "cleaned_cite_text": replaced_text,
#                 "citation_id": f"{row['CitingpaperID']}_{_}",
#                 "citation_excerpt_index": 0,
#                 "section_name": "introduction"
#             }
#             # 逐行写入文件（每行一个独立的JSON对象）
#             f.write(json.dumps(result, ensure_ascii=False) + '\n')

# print("save 'output.json'")

# import pandas as pd
# import re
# import json

# # 加载CSV文件
# df = pd.read_csv('./xlData/ARC_all_labeled.csv')

# # 定义一个函数来替换引用
# def replace_citation(text, citing_author_year):
#     # 提取作者和年份
#     authors_year = citing_author_year.split(',')
#     authors = authors_year[:-1]
#     year = authors_year[-1].strip()
    
#     # 为每个作者构建正则表达式
#     patterns = []
#     for author in authors:
#         # 提取姓氏
#         surname = author.split()[-1]
#         # 构建匹配姓氏部分字符的正则表达式
#         pattern = re.escape(surname[0].upper()) + re.escape(surname[1:]) + r'[A-Za-z\(\s,.]*?' + re.escape(year)
#         patterns.append(pattern)
    
#     # 组合多个作者的正则表达式
#     # 匹配任意一个作者的模式，并确保年份存在
#     combined_pattern = r'(?:' + r'|'.join(patterns) + r')'

#     match = re.search(combined_pattern, text)

#     if match:
#         # 替换为@@CITATION
#         replaced_text = re.sub(combined_pattern, '@@CITATION', text, count=1)
        
#         # 提取带@@CITATION的句子
#         citation_sentence_pattern = r'([^.]*?@@CITATION[^.]*\.)'
#         citation_sentence = re.search(citation_sentence_pattern, replaced_text)
#         if citation_sentence:
#             citation_text = citation_sentence.group(0)  # 提取包含@@CITATION的句子
#         else:
#             citation_text = ""

#         # 获取实际的引用文本（如 "Szubert et al., 2018"）
#         actual_citation = ', '.join(authors) + f', {year}'

#         # 替换 @@CITATION 为实际的引用
#         text_with_actual_citation = re.sub(r'@@CITATION', actual_citation, citation_text)

#         return text_with_actual_citation, citation_text, True
#     else: 
#         return text, text, False

# # 应用替换函数到每一行
# # df['CitationContent'] = df.apply(lambda row: replace_citation(row['CitationContent'], row['CitedAuthorYear']), axis=1)

# # 构建JSON结构
# # 打开文件，准备逐行写入
# with open('output.jsonl', 'w', encoding='utf-8') as f:
#     for _, row in df.iterrows():
#         citation_text, replaced_text, is_matched = replace_citation(row['CitationContent'], row['CitedAuthorYear'])
#         if is_matched:
#             start_0 = replaced_text.find('@@CITATION')
#             if start_0 != -1:
#                 end_0 = start_0 + len('@@CITATION') - 1
#                 start_1 = start_0 + 1  
#                 end_1 = end_0 + 1
#                 cite_marker_offset = [start_1, end_1]
#             else:
#                 cite_marker_offset = [0, 0]  
#             result = {
#                 "text": citation_text,  # 只保留带有实际引用信息的句子
#                 "citing_paper_id": row['CitingpaperID'],
#                 "cited_paper_id": row['CitedpaperID'],
#                 "citing_paper_year": int(row['CitingAuthorYear'].split(',')[-1].strip()),
#                 "cited_paper_year": int(row['CitedAuthorYear'].split(',')[-1].strip()),
#                 "citing_paper_title": row['CitingTitleAbstract'],
#                 "cited_paper_title": row['CitedTitleAbstract'],
#                 "cited_author_ids": row['CitedAuthorYear'].split(',')[:-1],
#                 "citing_author_ids": row['CitingAuthorYear'].split(',')[:-1],
#                 "extended_context": row['CitationContent'],  # 保持@@CITATION的原文
#                 "cleaned_cite_text": replaced_text,  # 保持@@CITATION的原文
#                 "section_number": None,
#                 "section_title": None,
#                 "intent": row['LabelType'],
#                 "cite_marker_offset": cite_marker_offset,
#                 "citation_id": f"{row['CitingpaperID']}_{_}",
#                 "citation_excerpt_index": 0,
#                 "section_name": "introduction"
#             }
#             # 逐行写入文件（每行一个独立的JSON对象）
#             f.write(json.dumps(result, ensure_ascii=False) + '\n')

# print("save 'output.jsonl'")

import pandas as pd
import re
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Process citation data and save to JSONL")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return

    df = pd.read_csv(args.input, encoding='utf-8-sig')

    def replace_citation(text, citing_author_year):
        try:
            # Splitting authors and year
            authors_year = citing_author_year.replace('(', ',').replace(')', '')
            authors_year = authors_year.split(',')
            authors = authors_year[:-1]
            year_part = authors_year[-1].strip()

            # Process year part to handle parentheses and extra spaces
            year_part = year_part.replace('(', '').replace(')', '')
            year = ''.join([c for c in year_part if c.isdigit()])
            if not year.isdigit():
                year = ''


            # Validate authors and year
            print(authors)
            print(year)
            authors = [author.strip() for author in authors if author.strip() != '']
            if not authors or not year:
                return text, text, False

            # 继续处理逻辑...

            # ... [略过其他逻辑，保持不变]
            
            # 调用原始函数逻辑

            # 构建每个作者的正则表达式模式
            patterns = []
            for author in authors:
                surname = author.split()[-1]
                pattern = re.escape(surname[0].upper()) + re.escape(surname[1:]) + r'[A-Za-z\(\s,.]*?' + re.escape(year)
                patterns.append(pattern)

            combined_pattern = r'(?:' + r'|'.join(patterns) + r')'

            match = re.search(combined_pattern, text)
            
            if match:
                replaced_text = re.sub(combined_pattern, '@@CITATION', text, count=1)
                
                citation_sentence_pattern = r'([^.]*?@@CITATION[^.]*\.)'
                citation_sentence = re.search(citation_sentence_pattern, replaced_text)
                if citation_sentence:
                    citation_text = citation_sentence.group(0).strip()
                else:
                    citation_text = ""
                
                actual_citation = ', '.join(authors) + f', {year}'
                text_with_actual_citation = re.sub(r'@@CITATION', actual_citation, citation_text)
                # return text_with_actual_citation, citation_text, True
                return text_with_actual_citation, replaced_text, True
            else:
                return text, text, False

        except Exception as e:
            print(f"Error processing {citing_author_year}: {e}")
            return text, text, False

    with open(args.output, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            try:
                citation_content = row['CitationContent']
                cited_author_year = row['CitedAuthorYear']

                # Call replace_citation and handle errors
                citation_text, replaced_text, is_matched = replace_citation(citation_content, cited_author_year)

                # ... 构建 JSON 对象并写入文件的逻辑保持不变...

                # Calculate offset
                start_0 = replaced_text.find('@@CITATION')
                if start_0 != -1:
                    end_0 = start_0 + len('@@CITATION') - 1
                    start_1 = start_0 + 1
                    end_1 = end_0 + 1
                    cite_marker_offset = [start_1, end_1]
                else:
                    cite_marker_offset = [0, 0]

                # Build JSON object
                result = {
                    "text": citation_text.strip(),
                    "citing_paper_id": row['CitingpaperID'],
                    "cited_paper_id": row['CitedpaperID'],
                    "citing_paper_year": int(row['CitingAuthorYear'].replace('(', ',').replace(')', '').split(',')[-1].strip()),
                    "cited_paper_year": int(row['CitedAuthorYear'].replace('(', ',').replace(')', '').split(',')[-1].strip()),
                    "cited_author_ids": [author.strip() for author in cited_author_year.split(',')[:-1]],
                    "citing_author_ids": [author.strip() for author in row['CitingAuthorYear'].split(',')[:-1]],
                    "extended_context": citation_content,
                    "cleaned_cite_text": replaced_text,
                    "section_number": None,
                    "section_title": None,
                    "intent": row['LabelType'],
                    "cite_marker_offset": cite_marker_offset,
                    "citation_id": f"{row['CitingpaperID']}_{index}",
                    "citation_excerpt_index": 0,
                    "section_name": "introduction"
                }

                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')

            except Exception as e:
                print(f"Skipping row {index} due to error: {e}")

    print(f"Successfully saved '{args.output}'")

if __name__ == "__main__":
    main()