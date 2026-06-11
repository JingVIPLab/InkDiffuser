#!/bin/bash

# FontDiffuser 第二阶段优化训练脚本
# 专注于风格学习，解决指标下降问题

CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed=123 \
    --experience_name="FontDiffuser_training_phase_2_optimized" \
    --data_root="data_examples" \
    --output_dir="outputs/FontDiffuser_phase2_style_focused" \
    --report_to="tensorboard" \
    --phase_2 \
    --phase_1_ckpt_dir="/home/skc/FontDiffuser/weights/FontDiffuser_double/checkpoint-280000" \
    --scr_ckpt_path="ckpt/scr_210000.pth" \
    \
    # 🎯 优化的风格学习参数
    --sc_coefficient=0.05 \
    --num_neg=16 \
    --nce_layers='0,1,2,3' \
    --temperature=0.07 \
    \
    # 📐 图像尺寸设置
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --content_encoder_downsample_size=3 \
    \
    # 🏗️ 模型架构
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    \
    # 🎓 优化的训练参数
    --train_batch_size=12 \
    --learning_rate=5e-5 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=2000 \
    --max_train_steps=50000 \
    --gradient_accumulation_steps=1 \
    \
    # 📊 损失权重调整
    --perceptual_coefficient=0.008 \
    --offset_coefficient=0.3 \
    \
    # 🔧 训练设置
    --ckpt_interval=5000 \
    --log_interval=25 \
    --drop_prob=0.05 \
    --mixed_precision="fp16" \
    --max_grad_norm=0.5 \
    \
    # ❌ 禁用非必要功能（专注风格）
    --use_high_freq=False \
    --use_intelligent_fusion=False \
    --use_dis_loss=False \
    \
    # 📝 注释说明：
    # sc_coefficient增加到0.05，提高风格损失权重
    # learning_rate提升到5e-5，确保有效学习
    # 使用cosine学习率调度，更平滑的收敛
    # max_train_steps增加到50000，充分训练
    # perceptual_coefficient略微降低，平衡风格和内容
    # offset_coefficient降低，减少对特征对齐的约束
    # drop_prob降低到0.05，减少随机性
    # 启用混合精度训练，提高效率
