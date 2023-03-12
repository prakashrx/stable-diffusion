from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import random

# print(torch.cuda.is_available())

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
# loop until 'Q' is pressed
while True:
    prompt = input("prompt: ")
    if prompt == "Q":
        break
    image = pipe(prompt).images[0]
    # generate a random image filename, every time
    filename = f"output_{random.randint(0, 1000000)}.png"
    image.save(filename)