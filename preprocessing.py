import argparse
import os
import nltk
from . import get_data, clean_data, tokenize_data

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="input dir")
parser.add_argument("output_dir", help="output dir")
args = parser.parse_args()

if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

data = get_data(args.input_dir)
print(len(data['TEXT_li']))
data["TEXT_li"] = [(tokenize_data(clean_data(e))) for e in data['TEXT_li']]
for k, v in data.items():
    with open(os.path.join(args.output_dir, k+'2.txt'), 'w') as f:
        for e in v:
            f.write(e.encode('utf-8') + '\n')

