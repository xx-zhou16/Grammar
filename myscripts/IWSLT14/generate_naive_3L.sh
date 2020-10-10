DATA_DIR=/mnt/xiangxin2/data/iwslt14/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive_3L
RESULT_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive_3L/generated

echo "Using GPU"$1

CUDA_VISIBLE_DEVICES=$1 fairseq-generate ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --gen-subset train \
    --results-path ${RESULT_DIR}
