exp_name=distill_2L_from_6L
DATA_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/naive_6L/generated/preprocessed
SAVE_DIR=/mnt/xiangxin2/data/wmt14/checkpoints/${exp_name}

CUDA_VISIBLE_DEVICES=0,1 fairseq-train ${DATA_DIR} \
  --arch transformer_wmt_en_de --share-all-embeddings \
  --encoder-layers 2 --decoder-layers 2 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0007 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
  --max-tokens 4096 --save-dir ${SAVE_DIR} \
  --update-freq 2 \
  --tensorboard-logdir ${SAVE_DIR} \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

# --log-interval 50
# --log-format json
# --no-progress-bar
# --save-interval-updates  1000
# --keep-interval-updates 20
