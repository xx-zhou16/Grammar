import os 
import generate_samples

exp_name = "top-2L-layerdrop-0.3_6L"
checkpoint_name = "checkpoint_best"
gpu_id = "0"

root_dir = "/mnt/xiangxin2/data/wmt14/checkpoints/"
exp_dir = os.path.join(root_dir, exp_name)
output_dir = os.path.join(exp_dir, checkpoint_name)
result_dir = 


script = ""
script += "exp_name={}\n".format(exp_name)
script += "CHECKPOINT_NAME={}\n".format(checkpoint_name)
script += "GPU_ID={}\n".format(gpu_id)



script += """
# exp_name=top-2L-layerdrop-0.3_6L
# CHECKPOINT_NAME=checkpoint_best
DATA_DIR=/mnt/xiangxin2/data/wmt14/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}
CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt
RESULT_DIR=${SAVE_DIR}/${CHECKPOINT_NAME}

CUDA_VISIBLE_DEVICES=${GPU_ID} fairseq-generate ${DATA_DIR} \\
    --path ${CHECKPOINT_PATH} \\
    --batch-size 128 --beam 5 \\
    --gen-subset valid \\
    --results-path ${RESULT_DIR} \\
    --model-overrides "{'encoder_layers_to_keep':'0,1,2,3,5', 'decoder_layers_to_keep':'0,1,2,3,5'}"
"""

import subprocess
#process = subprocess.Popen(script, shell=True, stdout=subprocess.PIPE)
process = subprocess.Popen(script, shell=True, stdout=open(os.devnull, 'wb'), stderr=subprocess.STDOUT)
process.wait()




