exp_name=naive_4L
DATA_DIR=/mnt/xiangxin2/data/wmt14/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}
RESULT_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}/generated

CUDA_VISBLE_DEVICES=1 fairseq-generate ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --gen-subset train \
    --results-path ${RESULT_DIR}
