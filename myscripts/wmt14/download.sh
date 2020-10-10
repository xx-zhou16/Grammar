# Download and prepare the data
cd examples/translation/
# WMT'17 data:
# bash prepare-wmt14en2de.sh
# or to use WMT'14 data:
bash prepare-wmt14en2de.sh --icml17
cd ../..

## Binarize the dataset
#TEXT=examples/translation/wmt17_en_de
#DATA_DIR=/mnt/xiangxin2/data/wmt14/preprocessed

#fairseq-preprocess \
#    --source-lang en --target-lang de \
#    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#    --destdir ${DATA_DIR} --thresholdtgt 0 --thresholdsrc 0 \
#    --workers 20