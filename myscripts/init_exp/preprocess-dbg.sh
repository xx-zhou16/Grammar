#DATA_DIR=/mnt/v-xizh46/lm/distill/random_truncated_train/raw_split
DATA_DIR=/mnt/xiangxin/dbg_split
#OUTPUT_DIR=/mnt/v-xizh46/lm/distill/random_truncated_train
OUTPUT_DIR=/mnt/xiangxin2/data/1bw-lm-dbg

fairseq-preprocess \
    --only-source \
    --srcdict /mnt/xiangxin/gpt2_vocab.fairseq \
    --trainpref ${DATA_DIR}/train.txt \
    --destdir ${OUTPUT_DIR}/bin-joined-dictionary \
    --joined-dictionary \
    --workers 20
