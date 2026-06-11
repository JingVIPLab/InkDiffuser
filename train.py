import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 EnhancedInkLoss,
                 DifferentiableInkStructureLoss,
                 create_dis_loss_preset,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr,
                 build_high_freq_encoder,
                 sobel_filter)
# 导入数值稳定版本的DIS损失
from src.modules.dis_loss_stable import DifferentiableInkStructureLossStable, create_dis_loss_preset_stable
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)


logger = get_logger(__name__)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():

    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Ser training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    # 构建高频编码器 - 只有在启用高频特征时才创建
    high_freq_encoder = None
    if args.use_high_freq:
        high_freq_encoder = build_high_freq_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))
        
        # 加载高频编码器权重（如果存在且启用）
        if args.use_high_freq and high_freq_encoder is not None:
            high_freq_encoder_path = f"{args.phase_1_ckpt_dir}/high_freq_encoder.pth"
            if os.path.exists(high_freq_encoder_path):
                high_freq_encoder.load_state_dict(torch.load(high_freq_encoder_path))
                logger.info("✅ 已加载高频编码器权重")
            else:
                logger.warning("⚠️  未找到高频编码器权重，使用随机初始化")
        

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        high_freq_encoder=high_freq_encoder,
        high_freq_fusion_type=getattr(args, 'high_freq_fusion_type', 'adaptive'),
        use_intelligent_fusion=getattr(args, 'use_intelligent_fusion', True)
        )

    # 💡 新增：在阶段二加载可学习的高频权重和融合模块
    if args.phase_2:
        # 只有在启用高频特征时才处理高频相关权重
        if args.use_high_freq and hasattr(model, 'learnable_high_freq_weight'):
            # 尝试加载阶段一学习的权重，如果没有则使用初始值
            learnable_weight_path = os.path.join(args.phase_1_ckpt_dir, "learnable_high_freq_weight.pth")
            if os.path.exists(learnable_weight_path):
                learned_weight = torch.load(learnable_weight_path)
                model.learnable_high_freq_weight.data = learned_weight
                logger.info(f"✅ 已加载阶段一学习的高频权重: {model.learnable_high_freq_weight.item():.6f}")
            else:
                initial_weight = 0.2
                model.learnable_high_freq_weight.data = torch.tensor(initial_weight)
                logger.info(f"⚠️  使用初始高频权重: {model.learnable_high_freq_weight.item():.6f}")
        
            # 加载高频融合模块权重
            fusion_weight_path = os.path.join(args.phase_1_ckpt_dir, "content_high_freq_fusion.pth")
            if os.path.exists(fusion_weight_path) and hasattr(model, 'content_high_freq_fusion'):
                model.content_high_freq_fusion.load_state_dict(torch.load(fusion_weight_path))
                logger.info(f"✅ 已加载高频融合模块权重，包含 {len(model.content_high_freq_fusion)} 个融合层")
            else:
                if hasattr(model, 'content_high_freq_fusion'):
                    logger.warning(f"⚠️  未找到高频融合模块权重文件，使用初始权重")
        elif args.use_high_freq:
            logger.warning("⚠️  启用了高频特征但模型不支持可学习高频权重")
        else:
            logger.info("ℹ️  第二阶段未启用高频特征，跳过高频权重加载")

    # Build content perceptaual Loss
    perceptual_loss = ContentPerceptualLoss()
    
    # Build 数值稳定版本的 DIS Loss if enabled (仅在第一阶段)
    dis_loss_fn = None
    enhanced_loss = None
    if getattr(args, 'use_dis_loss', False) and not args.phase_2:
        logger.info("🔧 第一阶段：使用数值稳定版本的DIS损失，避免数值爆炸")
        
        # 使用数值稳定版本替代原版本
        if hasattr(args, 'dis_preset'):
            dis_loss_fn = create_dis_loss_preset_stable(args.dis_preset)
            logger.info(f"✅ 启用数值稳定DIS墨迹扩散损失，预设配置: {args.dis_preset}")
        else:
            dis_loss_fn = DifferentiableInkStructureLossStable()
            logger.info("✅ 启用数值稳定DIS墨迹扩散损失，使用默认配置")

        # 创建增强损失函数（结合感知损失和DIS损失）
        enhanced_loss = EnhancedInkLoss(
            use_perceptual=True,
            use_dis=True,
            dis_preset=getattr(args, 'dis_preset', 'balanced'),
            perceptual_weight=args.perceptual_coefficient / 0.01,  # 标准化权重
            dis_weight=getattr(args, 'dis_loss_weight', 0.001)  # 使用更小的默认权重
        )
        
        # 替换DIS损失为稳定版本
        if hasattr(enhanced_loss, 'dis_loss'):
            enhanced_loss.dis_loss = dis_loss_fn
            
        logger.info(f"⚡ DIS损失权重: {getattr(args, 'dis_loss_weight', 0.001)} (已优化为数值稳定)")
    elif args.phase_2:
        logger.info("🎯 第二阶段：专注于风格学习，跳过DIS损失初始化")

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args)
        scr_checkpoint = torch.load(args.scr_ckpt_path)
        scr.load_state_dict(scr_checkpoint)
        scr.requires_grad_(False)
        
        # 🔍 SCR模块状态检查
        logger.info("🔍 SCR模块状态检查:")
        logger.info(f"  SCR检查点路径: {args.scr_ckpt_path}")
        logger.info(f"  SCR温度参数: {args.temperature}")
        logger.info(f"  NCE层配置: {args.nce_layers}")
        logger.info(f"  负样本数量: {args.num_neg}")
        logger.info(f"  SCR损失系数: {args.sc_coefficient}")
        
        # 检查SCR模块参数
        scr_params = sum(p.numel() for p in scr.parameters())
        logger.info(f"  SCR模块参数数量: {scr_params:,}")
        
        # 验证SCR模块是否正确冻结
        trainable_scr_params = sum(p.numel() for p in scr.parameters() if p.requires_grad)
        if trainable_scr_params > 0:
            logger.warning(f"⚠️  SCR模块有{trainable_scr_params:,}个可训练参数，这可能不是预期的")
        else:
            logger.info("✅ SCR模块已正确冻结")

    # Load the datasets
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN(),num_workers=4,pin_memory=True)
    
    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    
    # 💡 新增：显示训练参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 模型参数统计:")
    logger.info(f"  总参数数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    # 显示各组件的参数数量
    if hasattr(model, 'unet'):
        unet_params = sum(p.numel() for p in model.unet.parameters())
        logger.info(f"  UNet参数: {unet_params:,}")
    
    if hasattr(model, 'style_encoder'):
        style_params = sum(p.numel() for p in model.style_encoder.parameters())
        logger.info(f"  Style Encoder参数: {style_params:,}")
    
    if hasattr(model, 'content_encoder'):
        content_params = sum(p.numel() for p in model.content_encoder.parameters())
        logger.info(f"  Content Encoder参数: {content_params:,}")
    
    if args.use_high_freq:
        if hasattr(model, 'high_freq_encoder') and model.high_freq_encoder is not None:
            high_freq_params = sum(p.numel() for p in model.high_freq_encoder.parameters())
            logger.info(f"  High Freq Encoder参数: {high_freq_params:,}")
        
        if hasattr(model, 'content_high_freq_fusion') and model.content_high_freq_fusion:
            fusion_params = sum(p.numel() for p in model.content_high_freq_fusion.parameters())
            logger.info(f"  🎯 High Freq Fusion参数: {fusion_params:,}")
            logger.info(f"     融合模块数量: {len(model.content_high_freq_fusion)}")
        
        if hasattr(model, 'learnable_high_freq_weight'):
            logger.info(f"  🎛️ 可学习高频权重: {model.learnable_high_freq_weight.item():.6f}")
    else:
        logger.info("  ℹ️  未启用高频特征，跳过高频相关参数统计")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    ## move scr module to the target deivces
    if args.phase_2:
        scr = scr.to(accelerator.device)
    
    # 移动DIS损失函数到目标设备
    if enhanced_loss is not None:
        enhanced_loss = enhanced_loss.to(accelerator.device)
    if dis_loss_fn is not None:
        dis_loss_fn = dis_loss_fn.to(accelerator.device)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            
            # Phase 2 SCR准备：获取负样本
            neg_images = None
            if args.phase_2:
                if "neg_images" in samples:
                    neg_images = samples["neg_images"].to(accelerator.device)
                else:
                    logger.warning("🚨 第二阶段训练但数据中未找到neg_images字段")
            
            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value==1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict the noise residual and compute loss
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    use_high_freq=args.use_high_freq)
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                
                # 数值稳定性检查 - 在使用之前检查是否有NaN/Inf
                if torch.isnan(norm_pred_ori).any() or torch.isinf(norm_pred_ori).any():
                    logger.warning("🚨 检测到norm_pred_ori中有NaN或Inf值，跳过这个batch")
                    continue
                if torch.isnan(norm_target_ori).any() or torch.isinf(norm_target_ori).any():
                    logger.warning("🚨 检测到norm_target_ori中有NaN或Inf值，跳过这个batch")
                    continue
                
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                
                # 检查感知损失是否有NaN
                if torch.isnan(percep_loss) or torch.isinf(percep_loss):
                    logger.warning("🚨 感知损失包含NaN或Inf，设置为0")
                    percep_loss = torch.tensor(0.0, device=target_images.device)
                
                # 基础损失计算
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss
                
                # 计算加权后的损失组件用于日志记录
                weighted_percep_loss = args.perceptual_coefficient * percep_loss
                weighted_offset_loss = args.offset_coefficient * offset_loss
                weighted_dis_loss = torch.tensor(0.0, device=target_images.device)
                weighted_scr_loss = torch.tensor(0.0, device=target_images.device)
                
                # Phase 2: 计算SCR损失
                if args.phase_2 and neg_images is not None:
                    try:
                        # 数值稳定性检查输入数据
                        if (torch.isnan(pred_original_sample_norm).any() or torch.isinf(pred_original_sample_norm).any() or
                            torch.isnan(target_images).any() or torch.isinf(target_images).any() or
                            torch.isnan(neg_images).any() or torch.isinf(neg_images).any()):
                            logger.warning("🚨 SCR输入数据包含NaN或Inf，跳过SCR损失计算")
                            weighted_scr_loss = torch.tensor(0.0, device=target_images.device)
                        else:
                            # 确保输入数据在合理范围内
                            pred_clamp = torch.clamp(pred_original_sample_norm, -1.0, 1.0)
                            target_clamp = torch.clamp(target_images, -1.0, 1.0)
                            neg_clamp = torch.clamp(neg_images, -1.0, 1.0)
                            
                            sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                                pred_clamp,          # 生成的图像（裁剪后）
                                target_clamp,        # 正样本（目标图像，裁剪后）
                                neg_clamp,           # 负样本（裁剪后）
                                nce_layers=args.nce_layers
                            )
                            
                            # 计算NCE损失
                            scr_loss = scr.calculate_nce_loss(
                                sample_s=sample_style_embeddings,
                                pos_s=pos_style_embeddings, 
                                neg_s=neg_style_embeddings
                            )
                            
                            # 数值稳定性检查SCR损失
                            if torch.isnan(scr_loss) or torch.isinf(scr_loss) or scr_loss < 0:
                                logger.warning("🚨 SCR损失包含NaN/Inf或为负值，设置为0")
                                scr_loss = torch.tensor(0.0, device=target_images.device)
                            
                            # 计算加权的SCR损失
                            weighted_scr_loss = args.sc_coefficient * scr_loss
                            
                            # 添加到总损失
                            loss += weighted_scr_loss
                            
                            if global_step % args.log_interval == 0:
                                # 详细的SCR损失分析
                                scr_raw = scr_loss.item()
                                scr_weighted = weighted_scr_loss.item()
                                scr_ratio = scr_weighted / loss.item() * 100 if loss.item() > 0 else 0
                                
                                logger.info(f"🎯 SCR损失详情: 原始={scr_raw:.6f}, 加权={scr_weighted:.6f}, "
                                          f"占总损失比例={scr_ratio:.2f}%, 系数={args.sc_coefficient}")
                                
                                # 检查SCR损失是否过小
                                if scr_weighted < 1e-6:
                                    logger.warning("⚠️  SCR损失过小，可能需要增加sc_coefficient")
                                elif scr_ratio < 1.0:
                                    logger.warning(f"⚠️  SCR损失占比过低({scr_ratio:.2f}%)，建议增加sc_coefficient")
                            
                    except Exception as e:
                        logger.warning(f"🚨 SCR损失计算出错: {e}，跳过这个batch的SCR损失")
                        weighted_scr_loss = torch.tensor(0.0, device=target_images.device)
                else:
                    # 如果不是第二阶段或没有负样本，确保SCR损失为0
                    if args.phase_2 and neg_images is None:
                        logger.warning("🚨 第二阶段训练但未提供负样本数据")
                    weighted_scr_loss = torch.tensor(0.0, device=target_images.device)
                
                # 第二阶段专注于风格学习，不使用DIS损失
                if not args.phase_2 and getattr(args, 'use_dis_loss', False) and dis_loss_fn is not None:
                    try:
                        dis_results = dis_loss_fn(
                            generated_img=norm_pred_ori,
                            target_img=norm_target_ori,
                            return_detailed=True)
                        
                        # 获取DIS损失权重
                        dis_weight = getattr(args, 'dis_loss_weight', 0.001)
                        
                        dis_loss_value = dis_results['dis_loss']
                        
                        # 数值稳定性检查DIS损失
                        if torch.isnan(dis_loss_value) or torch.isinf(dis_loss_value):
                            logger.warning("🚨 DIS损失包含NaN或Inf，设置为0")
                            dis_loss_value = torch.tensor(0.0, device=target_images.device)
                        
                        # 计算加权的DIS损失
                        weighted_dis_loss = dis_weight * dis_loss_value
                        
                        # 添加到总损失
                        loss += weighted_dis_loss
                        
                        # 记录DIS损失组件
                        if global_step % args.log_interval == 0:
                            for loss_name, loss_value in dis_results.items():
                                if torch.is_tensor(loss_value) and not torch.isnan(loss_value) and not torch.isinf(loss_value):
                                    accelerator.log({f"dis_{loss_name}": loss_value.item()}, step=global_step)
                                    
                    except Exception as e:
                        logger.warning(f"🚨 DIS损失计算出错: {e}，跳过这个batch的DIS损失")
                elif args.phase_2:
                    logger.info("ℹ️  第二阶段专注于风格学习，已跳过DIS损失计算") if global_step % (args.log_interval * 10) == 0 else None
                
                # 最终检查总损失是否有NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"🚨 总损失包含NaN或Inf: {loss}，跳过这个batch")
                    continue
                
                # We need to keep track of how many items we have processed in each epoch
                # to correctly handle gradient accumulation and ensure consistency across distributed training
                train_loss += loss.detach().float() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                
                # 增强梯度裁剪，防止梯度爆炸
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    
                    # 记录各个损失组件到tensorboard
                    accelerator.log({
                        "loss/diff_loss": diff_loss.item(),
                        "loss/weighted_percep_loss": weighted_percep_loss.item(),
                        "loss/weighted_offset_loss": weighted_offset_loss.item(),
                        "loss/weighted_dis_loss": weighted_dis_loss.item(),
                        "loss/weighted_scr_loss": weighted_scr_loss.item(),
                        "loss/total_loss": loss.item()
                    }, step=global_step)

                    if global_step % args.log_interval == 0:
                        # 获取当前高频权重信息
                        unwrapped_model = accelerator.unwrap_model(model)
                        current_high_freq_weight = "N/A (未启用)"
                        if args.use_high_freq and hasattr(unwrapped_model, 'learnable_high_freq_weight'):
                            current_high_freq_weight = f"{unwrapped_model.learnable_high_freq_weight.item():.6f}"
                        
                        # 详细的损失日志打印
                        log_msg = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Global Step {global_step} => "
                                  f"总损失 = {loss.item():.6f} | "
                                  f"扩散损失 = {diff_loss.item():.6f} | "
                                  f"感知损失(×{args.perceptual_coefficient}) = {weighted_percep_loss.item():.6f} | "
                                  f"偏移损失(×{args.offset_coefficient}) = {weighted_offset_loss.item():.6f}")
                        
                        # 只在第一阶段且DIS损失存在时添加到日志
                        if not args.phase_2 and weighted_dis_loss.item() > 0:
                            log_msg += f" | DIS损失(×{getattr(args, 'dis_loss_weight', 0.001)}) = {weighted_dis_loss.item():.6f}"
                        
                        if args.phase_2 and weighted_scr_loss.item() > 0:
                            log_msg += f" | SCR风格损失(×{args.sc_coefficient}) = {weighted_scr_loss.item():.6f}"
                        
                        if args.use_high_freq:
                            log_msg += f" | 高频权重 = {current_high_freq_weight}"
                        
                        # 第二阶段专门的标识
                        if args.phase_2:
                            log_msg += " | 🎨 风格专注模式"
                            
                        logger.info(log_msg)

                    if global_step % args.ckpt_interval == 0:
                        if accelerator.is_main_process:
                            if args.phase_2:
                                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"保存检查点到 {save_path}")
                            
                            model_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(model_save_path, exist_ok=True)
                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model.unet.state_dict(), os.path.join(model_save_path, "unet.pth"))
                            torch.save(unwrapped_model.style_encoder.state_dict(), os.path.join(model_save_path, "style_encoder.pth"))
                            torch.save(unwrapped_model.content_encoder.state_dict(), os.path.join(model_save_path, "content_encoder.pth"))
                            
                            # Save high_freq_encoder if it exists and is enabled
                            if args.use_high_freq and unwrapped_model.high_freq_encoder is not None:
                                torch.save(unwrapped_model.high_freq_encoder.state_dict(), os.path.join(model_save_path, "high_freq_encoder.pth"))
                                logger.info("已保存高频编码器权重")

                            # 💡 新增：保存可学习的高频权重和融合模块（仅当启用高频特征时）
                            if args.use_high_freq:
                                if hasattr(unwrapped_model, 'learnable_high_freq_weight'):
                                    torch.save(unwrapped_model.learnable_high_freq_weight, os.path.join(model_save_path, "learnable_high_freq_weight.pth"))
                                    logger.info(f"已保存可学习高频权重: {unwrapped_model.learnable_high_freq_weight.item():.6f}")

                                # 保存高频融合模块的权重
                                if hasattr(unwrapped_model, 'content_high_freq_fusion') and unwrapped_model.content_high_freq_fusion:
                                    torch.save(unwrapped_model.content_high_freq_fusion.state_dict(), os.path.join(model_save_path, "content_high_freq_fusion.pth"))
                                    logger.info(f"已保存高频融合模块权重，包含 {len(unwrapped_model.content_high_freq_fusion)} 个融合层")
                                
                            logger.info(f"保存模型到 {model_save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                train_loss = 0.0

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()