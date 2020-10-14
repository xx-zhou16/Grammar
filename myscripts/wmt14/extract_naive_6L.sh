exp_name=naive_6L
RESULT_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}/generated
TARGET_ROOT=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}/generated/preprocessed
ORIGIN_ROOT=/home/zxx2020/projects/fairseq/examples/translation/wmt14_en_de # raw data

python extract.py --src ${RESULT_DIR}/generate-train.txt --tgt ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/valid.en ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/valid.de ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/test.en ${TARGET_ROOT}
cp ${ORIGIN_ROOT}/test.de ${TARGET_ROOT}

DATA_DIR=/mnt/xiangxin2/data/wmt14/preprocessed

SRC_DICT=${DATA_DIR}/dict.en.txt
TGT_DICT=${DATA_DIR}/dict.de.txt

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref ${TARGET_ROOT}/train --validpref ${TARGET_ROOT}/valid --testpref ${TARGET_ROOT}/test \
    --srcdict ${SRC_DICT} --tgtdict ${TGT_DICT} \
    --destdir ${TARGET_ROOT} \
    --workers 20