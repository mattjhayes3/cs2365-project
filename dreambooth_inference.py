import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel, LoraConfig

generator = [torch.Generator(device='mps').manual_seed(i) for i in range(6)]


def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="mps", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

for prior in [1]:
    #0.5, 0.75, 0, 0.25, 
    if prior == 0:
        path = "/Users/matthewhayes/Downloads/dreambooth_hand_output_no_prior"
    elif prior == 1:
        path = "/Users/matthewhayes/Downloads/dreambooth_hand_output"
    elif prior == 0.5:
        path = "/Users/matthewhayes/Downloads/dreambooth_hand_output_prior_0_5"
    else:
        path = f"/Users/matthewhayes/Downloads/dreambooth_hand_output_prior_0_{int(prior*100)}"
    for prompt in [
        # "a photo of sks hand",
        # "a photo of a woman holding up sks hand",
        # "a woman with sks hand",
        # "sks hand and a woman",
        # "sks hand on a woman",
        # "cross between sks hand and a woman",
        # "sks hand in egypt",
        # "a photo of sks hand in egypt",
        # "a photo of sks hand on a woman",
        "a photo of sks hand in a field",
    ]:
        pipe = get_lora_sd_pipeline(path, 
                                    base_model_name_or_path='runwayml/stable-diffusion-v1-5', 
                                    adapter_name="hand")
        for i, image in enumerate(pipe(prompt, 
                                       num_inference_steps=30, 
                                       guidance_scale=7.5,
                                       generator = generator,
                                       num_images_per_prompt=len(generator)).images):
            save_path = f"inference/dreambooth_hand_prior_{int(prior*100)}/{prompt.replace('_', ' ')}_{i}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)