exp_name=disitll_2L_from_4L
DATA_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/naive/generated/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/iwslt14/checkpoints/${exp_name}

CUDA_VISIBLE_DEVICES=1,2,3 fairseq-train \
    ${DATA_DIR} \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --encoder-layers 2 --decoder-layers 2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${SAVE_DIR}
