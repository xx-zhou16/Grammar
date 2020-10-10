DATA_DIR=/mnt/xiangxin2/data/iwslt14/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive
RESULT_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive/generated

fairseq-generate ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --gen-subset train \
    --results-path ${RESULT_DIR}

# backup
cp ${RESULT_DIR}/generate-train.txt ${RESULT_DIR}/generate-train.txt.bak