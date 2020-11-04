import os
import json
from collections import OrderedDict

import torch

gpt2_model_path = "/mnt/xiangxin2/models/gpt2-medium/gpt2-medium-pytorch_model.bin"
gpt2_model_dict = torch.load(gpt2_model_path)
my_checkpoint = torch.load('/mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/checkpoint1.pt')
my_model_dict = my_checkpoint['model']

gpt2_vocab_path = "/mnt/xiangxin2/models/gpt2-medium/gpt2-medium-vocab.json"

# TODO: token embedding
gpt2_vocab_json = json.load(open(gpt2_vocab_path), object_pairs_hook=OrderedDict)

my_model_vocab = None
with open("/mnt/xiangxin2/data/1bw-lm-dbg/bin-joined-dictionary/dict.txt", 'r') as f:
    my_model_vocab = list(map(lambda x: x.strip().split()[0], f.readlines()))

mapping_inds = []
for i in range(len(my_model_vocab)):
    mapping_inds.append(gpt2_vocab_json[my_model_vocab[i]])


my_model_dict['decoder.embed_tokens.weight'][4:, :] = gpt2_model_dict['wte.weight'][mapping_inds, :]
my_model_dict['decoder.output_projection.weight'][4:, :] = gpt2_model_dict['wte.weight'][mapping_inds, :]

my_model_dict['decoder.embed_positions.weight'][2:] = gpt2_model_dict['wpe.weight'][0:1024]


for i in range(0, 24):
    my_model_dict['decoder.layers.{}.self_attn.q_proj.weight'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_attn.weight'.format(str(i))][:, 0:1024]
    my_model_dict['decoder.layers.{}.self_attn.k_proj.weight'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_attn.weight'.format(str(i))][:, 1024:2048]
    my_model_dict['decoder.layers.{}.self_attn.v_proj.weight'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_attn.weight'.format(str(i))][:, 2048:3072]
    my_model_dict['decoder.layers.{}.self_attn.q_proj.bias'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_attn.bias'.format(str(i))][0:1024]
    my_model_dict['decoder.layers.{}.self_attn.k_proj.bias'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_attn.bias'.format(str(i))][1024:2048]
    my_model_dict['decoder.layers.{}.self_attn.v_proj.bias'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_attn.bias'.format(str(i))][2048:3072]
    my_model_dict['decoder.layers.{}.self_attn.out_proj.weight'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_proj.weight'.format(str(i))]
    my_model_dict['decoder.layers.{}.self_attn.out_proj.bias'.format(str(i))] = gpt2_model_dict['h.{}.attn.c_proj.bias'.format(str(i))]
    my_model_dict['decoder.layers.{}.self_attn_layer_norm.weight'.format(str(i))] = gpt2_model_dict['h.{}.ln_1.weight'.format(str(i))]
    my_model_dict['decoder.layers.{}.self_attn_layer_norm.bias'.format(str(i))] = gpt2_model_dict['h.{}.ln_1.bias'.format(str(i))]
    my_model_dict['decoder.layers.{}.fc2.weight'.format(str(i))] = gpt2_model_dict['h.{}.mlp.c_proj.weight'.format(str(i))].permute(1,0)
    my_model_dict['decoder.layers.{}.fc2.bias'.format(str(i))] = gpt2_model_dict['h.{}.mlp.c_proj.bias'.format(str(i))]
    my_model_dict['decoder.layers.{}.fc1.weight'.format(str(i))] = gpt2_model_dict['h.{}.mlp.c_fc.weight'.format(str(i))].permute(1,0)
    my_model_dict['decoder.layers.{}.fc1.bias'.format(str(i))] = gpt2_model_dict['h.{}.mlp.c_fc.bias'.format(str(i))]
    my_model_dict['decoder.layers.{}.final_layer_norm.weight'.format(str(i))] = gpt2_model_dict['h.{}.ln_2.weight'.format(str(i))]
    my_model_dict['decoder.layers.{}.final_layer_norm.bias'.format(str(i))] = gpt2_model_dict['h.{}.ln_2.bias'.format(str(i))]


my_model_dict['decoder.layer_norm.weight'] = gpt2_model_dict['ln_f.weight']
my_model_dict['decoder.layer_norm.bias'] = gpt2_model_dict['ln_f.bias']

torch.save(my_checkpoint, '/mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/gpt2-checkpoint.pt')
