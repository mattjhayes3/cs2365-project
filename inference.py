from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
import shutil

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                              #  torch_dtype=torch.float16,
                                               torch_dtype=torch.float,
                                               use_safetensors=True,
                                               safety_checker=None,
                                               )
# pipe.load_textual_inversion("diffusers/examples/textual_inversion/textual_inversion_handshake/")

for r in [4, 64]:
# for r in [64]:
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
  for steps in [5000, 10000, 15000, 20000, 25000, 30000]:
  # for steps in [30000]:
    pipe.load_lora_weights(f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/LoRAs/hands and palms r{r}/checkpoint-{steps}", weight_name="pytorch_model.bin")
    # pipe.load_lora_weights(f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/LoRAs/hands and palms r{r}/ch")
    # pipe.unet.load_attn_procs(f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/LoRAs/hands and palms r{r}/")
    # pipe.unet.load_attn_procs(f"/Users/matthewhayes/Downloads/pokemon checkpoint-8000", weight_name="pytorch_model.bin")
    pipe.to("mps")
    # pipe.unet.to("mps")
    # prompt = "a photo of a <handshake>"
    # prompt = "A photo of the palm of a hand with five fingers spread out"
    # prompt = "a man holding up one hand"
    # prompt = "a man giving a thumbs up"
    # prompt = "a photo of a handshake"
    prompts = [
      # "back of fair-skinned male left hand",
      # "front of fair-skinned male left hand",
      # "back of fair-skinned female left hand",
      # "front of fair-skinned female left hand",
      #   "back of fair-skinned male right hand",
      # "front of fair-skinned male right hand",
      # "back of fair-skinned female right hand",
      "front of fair-skinned female right hand",
    ]
    # prompts = ['green eyes pokemon']
    for prompt in prompts:
      for num_steps in [30]:
          images = pipe(prompt, num_inference_steps=num_steps, 
                    num_images_per_prompt=6,
                    #   guidance_scale=15
                    # cross_attention_kwargs={"scale": 10.0}
                    ).images
          for i, image in enumerate(images):
              save_path = f'inference/hands and palms r{r}_lora/{prompt.replace(" ", "_")}_chkpt{steps}_steps{num_steps}_{i}.png'
              # save_path = f'inference/pokemmon_test2/{prompt.replace(" ", "_")}_chkpt{steps}_steps{num_steps}_{i}.png'
              os.makedirs(os.path.dirname(save_path), exist_ok=True)
              image.save(save_path)

# images = pipe(prompt, num_inference_steps=100, num_images_per_prompt=4).images
# for i, image in enumerate(images):
#     image.save(f'inference/handbaseline_100_{i}.png')

# images = pipe(prompt, num_inference_steps=200, num_images_per_prompt=4).images
# for i, image in enumerate(images):
#     image.save(f'inference/handbaseline_200_{i}.png')

# images = pipe(prompt, num_inference_steps=500, num_images_per_prompt=4).images
# for i, image in enumerate(images):
#     image.save(f'inference/handbaseline_500_{i}.png')