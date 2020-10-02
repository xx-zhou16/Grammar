import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description=("extract"))
parser.add_argument('--input_dir', type=str, help='/path/to/generate_{train/test}.txt')
parser.add_argument('--output_dir', type=str, help='/path/to/raw/')
args = parser.parse_args()
setattr(args, "src", os.path.join(args.output_dir, "train.src-trg.src"))
setattr(args, "trg", os.path.join(args.output_dir, "train.src-trg.trg"))

with open(args.input_dir, 'r') as input_file:
	lines = input_file.readlines()

def safe_index(toks, index, default):
	try:
		return toks[index]
	except IndexError:
		return default

print("src =", args.src)
print("trg =", args.trg)

with open(args.src, 'w') as src_file, open(args.trg, 'w') as trg_file:
	for line in tqdm(lines):
		line = line.strip()
		if line.startswith('S-'):
			src = safe_index(line.rstrip().split('\t'), 1, '')
		elif line.startswith('H-'):
			if src is not None:
				trg = safe_index(line.rstrip().split('\t'), 2, '')
				# if validate(src, tgt):
				# 	print(src, file=src_h)
				# 	print(tgt, file=tgt_h)
				src_file.write(src + '\n')
				trg_file.write(trg + '\n')
				src = None