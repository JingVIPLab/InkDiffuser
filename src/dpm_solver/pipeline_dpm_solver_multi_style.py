import torch
from PIL import Image

from .dpm_solver_pytorch import (NoiseScheduleVP, 
                                model_wrapper, 
                                DPM_Solver)

class FontDiffuserDPMPipeline():
    """FontDiffuser pipeline with DPM_Solver scheduler - 支持多风格模式.
    """
    
    def __init__(
        self, 
        model, 
        ddpm_train_scheduler,
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5
    ):
        super().__init__()
        self.model = model
        self.train_scheduler_betas = ddpm_train_scheduler.betas
        # Define the noise schedule
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.train_scheduler_betas)

        self.version = version
        self.model_type = model_type
        self.guidance_type = guidance_type
        self.guidance_scale = guidance_scale

    def numpy_to_pil(self, images):
        """Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def generate(
        self,
        content_images,
        style_images,
        batch_size,
        order,
        num_inference_step,
        content_encoder_downsample_size,
        t_start=None,
        t_end=None,
        dm_size=(96, 96),
        algorithm_type="dpmsolver++",
        skip_type="time_uniform",
        method="multistep",
        correcting_x0_fn=None,
        generator=None,
        use_high_freq=False,
        # 🌟 新增多风格支持参数
        style_images2=None,
        use_multi_style=False,
    ):
        model_kwargs = {}
        # 🌟 根据多风格模式选择版本标识
        if use_multi_style:
            model_kwargs["version"] = "V3_MULTI_STYLE"  # 新的多风格版本
        else:
            model_kwargs["version"] = self.version  # 原始版本
        model_kwargs["content_encoder_downsample_size"] = content_encoder_downsample_size
        model_kwargs["use_high_freq"] = use_high_freq
        model_kwargs["use_multi_style"] = use_multi_style  # 🌟 传递多风格标志

        # 🌟 多风格条件准备
        cond = []
        cond.append(content_images)
        cond.append(style_images)
        
        # 如果是多风格模式且有第二个风格图片，添加到条件中
        print(f"🔍 Pipeline检测: use_multi_style={use_multi_style}, style_images2 is None={style_images2 is None}")
        if style_images2 is not None:
            images_equal = torch.equal(style_images, style_images2)
            print(f"🔍 Pipeline: style_images与style_images2相等={images_equal}")
        
        if use_multi_style and style_images2 is not None and not torch.equal(style_images, style_images2):
            cond.append(style_images2)
            print(f"🎨 Pipeline: 真正的多风格模式，使用两张不同的风格图片")
        else:
            # 单风格模式或没有第二张风格图片时，复制第一张
            cond.append(style_images)
            if use_multi_style:
                print(f"⚠️  Pipeline: 多风格模式但只有一张风格图片，将复制使用")
            else:
                print(f"📝 Pipeline: 单风格模式，复制第一张风格图片")

        # 🌟 多风格无条件准备
        uncond = []
        # 直接使用输入张量的设备，确保设备一致性
        input_device = content_images.device
        uncond_content_images = torch.ones_like(content_images).to(input_device)
        uncond_style_images = torch.ones_like(style_images).to(input_device)
        uncond.append(uncond_content_images)
        uncond.append(uncond_style_images)
        
        # 为第二张风格图片准备无条件版本
        if use_multi_style and style_images2 is not None:
            uncond_style_images2 = torch.ones_like(style_images2).to(input_device)
            uncond.append(uncond_style_images2)
        else:
            uncond.append(uncond_style_images)  # 复制第一张风格图片的无条件版本

        # 2.Convert the discrete-time model to the continuous-time
        model_fn = model_wrapper(
            model=self.model,
            noise_schedule=self.noise_schedule,
            model_type=self.model_type,
            model_kwargs=model_kwargs,
            guidance_type=self.guidance_type,
            condition=cond, 
            unconditional_condition=uncond,
            guidance_scale=self.guidance_scale
        )

        # 3. Define dpm-solver and sample by multistep DPM-Solver.
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn
        )

        # 4. Generate
        # Sample gaussian noise to begin loop => [batch, 3, height, width]
        x_T = torch.randn(
            (batch_size, 3, dm_size[0], dm_size[1]),
            generator=generator,
        )
        x_T = x_T.to(input_device)

        x_sample = dpm_solver.sample(
            x=x_T,
            steps=num_inference_step,
            order=order,
            skip_type=skip_type,
            method=method,
        )

        x_sample = (x_sample / 2 + 0.5).clamp(0, 1)
        x_sample = x_sample.cpu().permute(0, 2, 3, 1).numpy()
    
        x_images = self.numpy_to_pil(x_sample)

        return x_images 