EXP_NAME=dbg_gpt2
#DATA_DIR=/mnt/xiangxin2/data/1bw-lm/bin-joined-dictionary
DATA_DIR=/mnt/xiangxin2/data/1bw-lm-dbg/bin-joined-dictionary
SAVE_DIR=/mnt/xiangxin2/checkpoints/init_exp/${EXP_NAME}

CUDA_VISIBLE_DEVICES=3 fairseq-train ${DATA_DIR} \
  --task language_modeling \
  --arch transformer_lm \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-07 \
  --sample-break-mode eos \
  --num-workers 20 \
  --max-tokens 8000 --update-freq 8 \
  --fp16 \
  --decoder-input-dim 1024 \
  --decoder-output-dim 1024 \
  --decoder-embed-dim 1024 \
  --decoder-ffn-embed-dim 4096 \
  --decoder-layers 6 \
  --decoder-attention-heads 16 \
  --share-decoder-input-output-embed \
  --decoder-learned-pos \
  --max-target-positions 512 \
  --no-decoder-final-norm

#  --dropout 0.1 \
