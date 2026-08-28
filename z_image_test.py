import os

from lightx2v import LightX2VPipeline

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pipe = LightX2VPipeline(
    model_path=os.path.join(ROOT_DIR, "models", "Z-Image-Turbo"),
    model_cls="z_image",
    task="t2i",
)

# int8 (not fp8): the RTX 3070 Ti is Ampere and has no FP8 tensor cores,
# and int8-torchao needs only the pure-pip `torchao` package (no native
# kernel build, unlike the fp8-sgl/vllm paths the model card suggests).
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt=os.path.join(ROOT_DIR, "models", "Z-Image-Turbo-Quantized", "z_image_turbo_int8.safetensors"),
    quant_scheme="int8-torchao",
)

pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",  # ["model", "block"]; block needed to fit 8GB VRAM
)

pipe.create_generator(
    attn_mode="torch_sdpa",  # flash_attn3 needs Hopper; sdpa ships with torch and needs no extra install
    aspect_ratio="16:9",
    infer_steps=9,
    guidance_scale=1,
)

seed = 42
prompt = "A lighthouse on a rocky cliff at dusk, cinematic lighting, ultra detailed, 4k"
negative_prompt = ""
save_result_path = os.path.join(ROOT_DIR, "z_image_output.png")

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)

print(f"Saved to {save_result_path}")
