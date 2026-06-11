#!/bin/bash

# SCR模块快速训练脚本（用于测试和快速验证）
# 训练步数较少，适合验证数据集兼容性

echo "⚡ 开始SCR模块快速训练（测试模式）..."

CUDA_VISIBLE_DEVICES=3 python train_scr.py \
    --seed=123 \
    --experience_name="SCR_quick_test" \
    --data_root="data_examples" \
    --output_dir="outputs/scr_quick_test" \
    --report_to="tensorboard" \
    --scr_only \
    --scr_lr=2e-4 \
    --scr_warmup_steps=1000 \
    --scr_max_steps=50000 \
    --scr_batch_size=8 \
    --scr_save_interval=10000 \
    --scr_log_interval=50 \
    --scr_eval_interval=2000 \
    --temperature=0.07 \
    --mode="training" \
    --nce_layers='0,1,2,3' \
    --num_neg=8 \
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --scr_image_size=96 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_weight_decay=0.01 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=1.0 \
    --gradient_accumulation_steps=2 \
    --mixed_precision="fp16"

echo "🎯 快速训练模式说明："
echo "  - 训练步数: 50K（相比完整版本的300K）"
echo "  - 批次大小: 8（减少显存占用）"
echo "  - 负样本数: 8（减少计算量）"
echo "  - 梯度累积: 2步（等效批次大小16）"
echo ""
echo "💡 如果快速训练效果良好，再使用完整版本 train_scr.sh 进行完整训练"
