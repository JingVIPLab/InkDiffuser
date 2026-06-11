#!/bin/bash

# SCR模块训练脚本
# 在您的数据集上训练Style Consistency Regularization模块

echo "🚀 开始训练SCR模块..."

CUDA_VISIBLE_DEVICES=1 python train_scr.py \
    --seed=123 \
    --experience_name="SCR_training_custom_dataset" \
    --data_root="data_examples" \
    --output_dir="outputs/scr_custom_2" \
    --report_to="tensorboard" \
    --scr_only \
    --scr_lr=1e-4 \
    --scr_warmup_steps=5000 \
    --scr_max_steps=300000 \
    --scr_batch_size=16 \
    --scr_save_interval=20000 \
    --scr_log_interval=100 \
    --scr_eval_interval=5000 \
    --temperature=0.07 \
    --mode="training" \
    --nce_layers='0,1,2,3' \
    --num_neg=16 \
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --scr_image_size=96 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_weight_decay=0.01 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=1.0 \
    --gradient_accumulation_steps=1 \
    --mixed_precision="fp16" \
    --lr_scheduler="cosine"

# 📝 训练说明：
# scr_lr: SCR模块的学习率，通常比主模型稍低
# scr_max_steps: 总训练步数，建议300K以上确保充分训练
# temperature: NCE损失的温度参数，影响对比学习的难度
# nce_layers: 使用哪些VGG层进行风格对比，'0,1,2,3'是常用配置
# num_neg: 负样本数量，影响对比学习效果
# scr_batch_size: 批次大小，需要足够大以提供充足的负样本

echo "📊 训练完成后，您可以在以下位置找到训练好的SCR模型："
echo "  - 最佳模型: outputs/scr_custom/scr_training/scr_best.pth"
echo "  - 最终模型: outputs/scr_custom/scr_training/scr_final.pth"
echo "  - 定期检查点: outputs/scr_custom/scr_training/scr_*.pth"
echo ""
echo "💡 使用训练好的SCR模型进行第二阶段训练时，请更新 --scr_ckpt_path 参数"
