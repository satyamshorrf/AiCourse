from diffusers import DiffusionPipeline
import torch

model_path_sdxl = ("../../Models/models--stabilityai--stable-diffusion-xl-base-1.0/"
                   "snapshots/462165984030d82259a11f4367a4eed129e94a7b")

base = DiffusionPipeline.from_pretrained(model_path_sdxl,
                                         torch_dtype=torch.float16,
                                         variant="fp16",
                                         use_safetensors=True)

base.enable_model_cpu_offload()
prompt = "A cute cat jumping over a fence, cartoonic, colorful style."
image = base(prompt=prompt, num_inference_steps=20, guidance_scale=4).images[0]
image.show()



