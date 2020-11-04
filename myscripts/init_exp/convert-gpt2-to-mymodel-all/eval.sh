DATA_DIR=/mnt/xiangxin2/data/1bw-lm/bin-joined-dictionary
CHECKPOINT=/mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/gpt2-checkpoint.pt
RESULTS_PATH=/mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/dbg-result

fairseq-eval-lm ${DATA_DIR} \
    --task language_modeling \
    --path ${CHECKPOINT} \
    --max-sentences 1 \
    --sample-break-mode 'eos'\
    --results-path ${RESULTS_PATH}