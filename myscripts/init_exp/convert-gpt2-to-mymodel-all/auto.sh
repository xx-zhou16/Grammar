# rm -rf /mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/*.pt && \
# bash train_dbg_gpt2.sh && \
CUDA_VISIBLE_DEVICES=2 python convert.py
# rm -rf /mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/converted-train/*.pt && \
# mkdir -p /mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/converted-train
# cp /mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/gpt2-checkpoint.pt /mnt/xiangxin2/checkpoints/init_exp/dbg_gpt2_12L/converted-train/checkpoint_last.pt && \
# bash train_converted_gpt2.sh