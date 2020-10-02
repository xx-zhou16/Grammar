#DATA_DIR=/mnt/xiangxin/correction_model/preprocessed_data
DATA_DIR=/mnt/xiangxin/seq2seq_char/preprocessed_data
#MODEL_DIR=/mnt/xiangxin/correction_model/checkpoints/checkpoint_best.pt
MODEL_DIR=/mnt/xiangxin/seq2seq_char/checkpoints/checkpoint_last.pt

RESULT_DIR=/mnt/xiangxin2/data/generated

CUDA_VISIBLE_DEVICES=2 fairseq-generate ${DATA_DIR} \
  --path ${MODEL_DIR} \
  --beam 4 \
  --lenpen 0.6 \
  --gen-subset train \
  --results-path ${RESULT_DIR}

