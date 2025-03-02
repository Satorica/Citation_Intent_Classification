import json
import re
import os

def load_data(path):

    data = []
    for x in open(path, "r"):
        data.append(json.loads(x))
    return data

file_name = ['dev.jsonl','test.jsonl','train.jsonl']
input_dir = './acl-arc'
output_dir = './dataset'

for file in file_name:

    output_path = os.path.join(output_dir, file)
    input_path = os.path.join(input_dir, file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile, \
        open(output_path, 'w', encoding='utf-8') as outfile:

        json_data = load_data(input_path)
        # print(json_data)
        for data in json_data:
            data.pop("sents_before")
            data.pop("sents_after")
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # print(outfile)