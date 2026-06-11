#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py \
    --seed=123 \
    --experience_name="FontDiffuser_training_phase_1" \
    --data_root="data_examples" \
    --output_dir="weights/FontDiffuser_double_0.04" \
    --report_to="tensorboard" \
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=16 \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=440000 \
    --ckpt_interval=40000 \
    --gradient_accumulation_steps=1 \
    --log_interval=100 \
    --learning_rate=1e-4 \
    --lr_scheduler="linear" \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision="fp16" \
    --use_high_freq=True \
    --use_intelligent_fusion=True \
    --high_freq_fusion_type="adaptive" \
    --fusion_channel_reduction=16 \
    --fusion_spatial_kernel=7 \
    --fusion_cross_attn_heads=8 \
    --fusion_time_emb_dim=512 \
    --use_dis_loss=True \
    --dis_loss_weight=0.04 \
    --dis_preset=balanced \
    --dis_adaptive_weights=True \
    --dis_temperature=1.5 \
    --dis_erosion_kernel_size=2 \
    --dis_dilation_kernel_size=4 



