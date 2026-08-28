import os
from datetime import datetime
import gradio as gr
from lightx2v import LightX2VPipeline

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(ROOT_DIR, "z_image_gradio_output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"]

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
    offload_granularity="block",  # needed to fit the 8GB VRAM budget
)

# create_generator() reloads the DIT/text-encoder weights from disk (~40s),
# so only re-run it when aspect ratio / steps / guidance actually change.
_last_generator_settings = None


def ensure_generator(aspect_ratio, infer_steps, guidance_scale):
    global _last_generator_settings
    settings = (aspect_ratio, infer_steps, guidance_scale)
    if settings != _last_generator_settings:
        pipe.create_generator(
            attn_mode="torch_sdpa",  # flash_attn3 needs Hopper; sdpa ships with torch, no extra install
            aspect_ratio=aspect_ratio,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
        )
        _last_generator_settings = settings


def generate(prompt, negative_prompt, seed, aspect_ratio, infer_steps, guidance_scale, use_custom_resolution, width, height):
    if not prompt:
        return None

    ensure_generator(aspect_ratio, int(infer_steps), guidance_scale)

    # target_shape is a per-call arg (unlike aspect_ratio, it doesn't require
    # a model reload); actual W/H get rounded down to a multiple of 16 and
    # clamped to [256, 1664] by the runner regardless of what we pass here.
    target_shape = [int(height), int(width)] if use_custom_resolution else []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_path = os.path.join(OUTPUT_PATH, f"{timestamp}.png")
    output_path = OUTPUT_PATH + f"_{timestamp}.png"
    # txt_path = os.path.join(OUTPUT_PATH, f"{timestamp}.txt")
    txt_path = OUTPUT_PATH + f"_{timestamp}.txt"
    pipe.generate(
        seed=int(seed),
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=output_path,
        target_shape=target_shape,
    )
    # Sidecar file with the settings used, so a later image can be reproduced
    with open(txt_path, "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Aspect Ratio: {aspect_ratio}\n")
        f.write(f"Use Custom Resolution: {use_custom_resolution}\n")
        f.write(f"Width: {width}\n")
        f.write(f"Height: {height}\n")
        f.write(f"Inference Steps: {infer_steps}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Seed: {seed}\n")
    return output_path


with gr.Blocks(title="Z-Image-Turbo (INT8)") as demo:
    gr.Markdown("# Z-Image-Turbo — INT8 Quantized Text-to-Image")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image you want...")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Things to avoid (optional)...")
            aspect_ratio = gr.Dropdown(ASPECT_RATIOS, value="16:9", label="Aspect Ratio")
            use_custom_resolution = gr.Checkbox(label="Use custom resolution (overrides aspect ratio)", value=False)
            width = gr.Slider(256, 1664, value=1664, step=16, label="Width")
            height = gr.Slider(256, 1664, value=928, step=16, label="Height")
            infer_steps = gr.Slider(1, 30, value=9, step=1, label="Inference Steps")
            guidance_scale = gr.Slider(0, 5, value=1, step=0.1, label="Guidance Scale (1 = CFG off, as the Turbo model was distilled for)")
            seed = gr.Number(label="Seed", value=42, precision=0)
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="Generated Image")

    submit_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, seed, aspect_ratio, infer_steps, guidance_scale, use_custom_resolution, width, height],
        outputs=image_output,
    )

if __name__ == "__main__":
    demo.launch()
