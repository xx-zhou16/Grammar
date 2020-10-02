DATA_DIR=/mnt/xiangxin/correction_model/preprocessed_data
SAVE_DIR=/mnt/xiangxin2/checkpoints/leven_grammar_cn

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --save-interval-updates 10000 \
    --max-update 300000
    #--apply-bert-init \
    #--ddp-backend=no_c10d \
