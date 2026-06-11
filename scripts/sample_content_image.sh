python sample.py \
    --ckpt_dir="/home/skc/FontDiffuser/weights/FontDiffuser_double/checkpoint-280000" \
    --content_image_path="/home/skc/FontDiffuser/data_examples/train/ContentImage/匾.jpg" \
    --style_image_path="/home/skc/FontDiffuser/data_examples/train/TargetImage/FZ-PCL-Font-Bold/FZ-PCL-Font-Bold+丐.jpg" \
    --save_image \
    --save_image_dir="outputs/11111" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"    

