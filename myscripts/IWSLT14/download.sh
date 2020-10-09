# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

DATA_DIR=/mnt/xiangxin2/data/iwslt14/preprocessed

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir ${DATA_DIR} \
    --workers 20