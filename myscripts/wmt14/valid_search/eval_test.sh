exp_name=top-2L-layerdrop-0.3_6L
DATA_DIR=/mnt/xiangxin2/data/wmt14/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}
CHECKPOINT_NAME=checkpoint_best
CHECKPOINT_PATH=${SAVE_DIR}/${CHECKPOINT_NAME}.pt
RESULT_DIR=${SAVE_DIR}/${CHECKPOINT_NAME}

CUDA_VISIBLE_DEVICES=1 fairseq-generate ${DATA_DIR} \
    --path ${CHECKPOINT_PATH} \
    --batch-size 128 --beam 5 \
    --gen-subset valid \
    --results-path ${RESULT_DIR} \
    --model-overrides "{'encoder_layers_to_keep':'0,1,2,3,5', 'decoder_layers_to_keep':'0,1,2,3,5'}"

