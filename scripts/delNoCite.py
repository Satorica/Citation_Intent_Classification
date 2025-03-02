import json
import argparse
import os

def filter_cited_entries(input_file, output_file):
    # 打开输入文件并读取JSONL数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 过滤符合条件的数据
    filtered_lines = []
    for line in lines:
        try:
            obj = json.loads(line)
            # 检查 'cleaned_cite_text' 是否存在且包含 '@@CITATION'
            if 'cleaned_cite_text' in obj and '@@CITATION' in obj['cleaned_cite_text']:
                filtered_lines.append(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    # 将过滤后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print(f"Filtered data saved to {output_file}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Filter JSONL entries based on @@CITATION")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return

    # 调用过滤函数
    filter_cited_entries(args.input, args.output)

if __name__ == "__main__":
    main()