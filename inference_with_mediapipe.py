from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
import shutil
from diffusers import DDPMScheduler, DDIMScheduler, KDPM2DiscreteScheduler
from PIL import Image
import numpy as np

outdir = 'inference/test'

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                              #  torch_dtype=torch.float16,
                                               torch_dtype=torch.float,
                                               use_safetensors=True,
                                               safety_checker=None,
                                               )
# pipe.to("mps")
# pipe.load_textual_inversion("diffusers/examples/textual_inversion/textual_inversion_handshake/")
def callback(pipe, step, timestep, kwargs):
    print(f"callback step={step}, timestep={timestep}")#, kwargs={kwargs}")
    latents = kwargs['latents']
    images = pipe.decode_latents(latents) * 255
    for image in range(images.shape[0]):
      Image.fromarray(np.uint8(images[image])).save(f"{outdir}/karras2_kdpm2d_short_img{image}_step{timestep}.png")
       
    
    return kwargs
    
generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(3)]
# for r in [4, 64]:
for r in [64]:
  # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
  pipe.scheduler.config.use_karras_sigmas = True
  for steps in [20000]: #[5000, 10000, 15000, 20000, 25000, 30000]:
  # for steps in [30000]:
    # pipe.load_lora_weights(f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/LoRAs/hands and palms r{r}/checkpoint-{steps}", weight_name="pytorch_model.bin")
    # pipe.load_lora_weights(f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/LoRAs/hands and palms r{r}/ch")
    # pipe.unet.load_attn_procs(f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/LoRAs/hands and palms r{r}/")
    # pipe.unet.load_attn_procs(f"/Users/matthewhayes/Downloads/pokemon checkpoint-8000", weight_name="pytorch_model.bin")
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
      "a photo of a hand",
    ]
    # prompts = ['green eyes pokemon']
    for prompt in prompts:
      for num_steps in [6]:
          images = pipe(prompt, num_inference_steps=num_steps, 
                    generator=generator,
                    num_images_per_prompt=len(generator),
                    callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=pipe._callback_tensor_inputs,
                    #   guidance_scale=15
                    # cross_attention_kwargs={"scale": 10.0}
                    # timesteps = [999, 966, 932, 899, 866, 832, 799, 766, 733, 699, 666, 633, 599, 566,
                    #              533, 499, 466, 433, 400, 366, 333, 300, 266, 233, 200, 166, 133, 100,
                    #              67,  33],
                    ).images
          for i, image in enumerate(images):
              save_path = f'{outdir}/{prompt.replace(" ", "_")}_chkpt{steps}_steps{num_steps}_{i}.png'
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