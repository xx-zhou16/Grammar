import os
import re

def read_BLEU4(file_path):
    last_line = None
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-1]
        target = re.search(r'BLEU4 = (.*?), ', last_line) 
        #print(target.group(1))
        target = target.group(1)
    return float(target)


if __name__ == "__main__":
    file_path = "/mnt/xiangxin2/data/wmt14/checkpoints/top-2L-layerdrop-0.3_6L/checkpoint_best/generate-valid.txt"
    print(read_BLEU4(file_path))
