#DATA_DIR=/mnt/v-xizh46/lm/distill/random_truncated_train/raw_split
DATA_DIR=/mnt/xiangxin/raw_split
#OUTPUT_DIR=/mnt/v-xizh46/lm/distill/random_truncated_train
OUTPUT_DIR=/mnt/xiangxin2/data/1bw-lm

fairseq-preprocess \
    --only-source \
    --srcdict /mnt/xiangxin/gpt2_vocab.fairseq \
    --trainpref ${DATA_DIR}/train.txt \
    --validpref ${DATA_DIR}/dev.txt \
    --testpref ${DATA_DIR}/test.txt \
    --destdir ${OUTPUT_DIR}/bin-joined-dictionary \
    --joined-dictionary \
    --workers 20
