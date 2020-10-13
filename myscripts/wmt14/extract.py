import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description=("extract"))
parser.add_argument('--src', type=str, help='/path/to/generate_{train/test}.txt')
parser.add_argument('--tgt_dir', type=str, help='/path/to/raw/')
args = parser.parse_args()
setattr(args, "tgt_en", os.path.join(args.tgt_dir, "train.en"))
setattr(args, "tgt_de", os.path.join(args.tgt_dir, "train.de"))

with open(args.src, 'r') as srcfile:
	lines = srcfile.readlines()

def safe_index(toks, index, default):
	try:
		return toks[index]
	except IndexError:
		return default

if not os.path.exists(args.tgt_dir):
	os.mkdir(args.tgt_dir)

with open(args.tgt_en, 'w') as tgt_en_file, open(args.tgt_de, 'w') as tgt_de_file:
	for line in tqdm(lines):
		line = line.strip()
		if line.startswith('S-'):
			tgt_en = safe_index(line.rstrip().split('\t'), 1, '')
		elif line.startswith('H-'):
			if tgt_en is not None:
				tgt_de = safe_index(line.rstrip().split('\t'), 2, '')
				# if validate(src, tgt):
				# 	print(src, file=src_h)
				# 	print(tgt, file=tgt_h)
				tgt_en_file.write(tgt_en + '\n')
				tgt_de_file.write(tgt_de + '\n')
				tgt_en = None