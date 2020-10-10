RESULT_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive/generated
TARGET_ROOT=/mnt/xiangxin2/data/iwslt14/checkpoints/naive/generated/raw
ORIGIN_ROOT=/home/zxx2020/projects/fairseq/examples/translation/iwslt14.tokenized.de-en # raw data
DEST_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive/generated/preprocessed
mkdir -p ${DEST_DIR}

python extract.py --src ${RESULT_DIR}/generate-train.txt --tgt ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/valid.en ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/valid.de ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/test.en ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/test.de ${TARGET_ROOT}

DATA_DIR=/mnt/xiangxin2/data/iwslt14/preprocessed

SRC_DICT=${DATA_DIR}/dict.de.txt
TGT_DICT=${DATA_DIR}/dict.en.txt

fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref ${TARGET_ROOT}/train --validpref ${TARGET_ROOT}/valid --testpref ${TARGET_ROOT}/test \
    --srcdict ${SRC_DICT} --tgtdict ${TGT_DICT} \
    --destdir ${DEST_DIR} \
    --workers 20