INPUT_DIR=/mnt/xiangxin2/data/generated/extracted
OUTPUT_DIR=/mnt/xiangxin2/data/generated/preprocessed
fairseq-preprocess \
    --source-lang src --target-lang trg \
    --trainpref ${INPUT_DIR}/train.src-trg \
    --tgtdict /mnt/xiangxin/seq2seq_char/preprocessed_data/dict.src.txt \
    --destdir ${OUTPUT_DIR} --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20