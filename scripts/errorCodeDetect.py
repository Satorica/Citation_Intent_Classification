ACL_TRAIN_PATH = "../acl-arc/train_xlData.jsonl"
def find_problematic_characters(file_path):
    try:
        with open(file_path, 'rb') as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    # 解码字节流为 utf-8
                    line.decode('utf-8')
                except UnicodeDecodeError as e:
                    print(f"Error at line {line_number}, position {e.start}: {e.reason}")
                    # 打印异常周围的内容
                    surrounding_start = max(0, e.start - 10)
                    surrounding_end = min(len(line), e.start + 10)
                    surrounding_bytes = line[surrounding_start:surrounding_end]
                    print(f"Surrounding bytes: {surrounding_bytes}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")

# 替换为你的文件路径
find_problematic_characters(ACL_TRAIN_PATH)