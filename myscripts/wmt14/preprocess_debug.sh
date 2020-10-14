## Download and prepare the data
#cd examples/translation/
## WMT'17 data:
## bash prepare-wmt14en2de.sh
## or to use WMT'14 data:
#bash prepare-wmt14en2de.sh --icml17
#cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_de_debug
DATA_DIR=/mnt/xiangxin2/data/wmt14/preprocessed_debug

SRC_DICT=/mnt/xiangxin2/data/wmt14/preprocessed/dict.en.txt

fairseq-preprocess \
    --source-lang en --target-lang de \
    --validpref $TEXT/valid \
    --joined-dictionary \
    --srcdict ${SRC_DICT} \
    --destdir ${DATA_DIR} --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
