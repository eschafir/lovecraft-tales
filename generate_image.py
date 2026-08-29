#!/usr/bin/env python3
"""
Z-Image-Turbo Generator CLI (runs in 'z-image' conda environment).
"""

import argparse
import os
import random
import sys
from lightx2v import LightX2VPipeline

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Generate image using Z-Image-Turbo.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--output", type=str, required=True, help="Output image file path.")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, deformed, text, watermark, signature", help="Negative prompt.")
    parser.add_argument("--aspect-ratio", type=str, default="1:1", choices=["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"], help="Aspect ratio.")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8).")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale (default: 1.0).")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random).")
    parser.add_argument("--width", type=int, default=768, help="Image width (default: 768).")
    parser.add_argument("--height", type=int, default=768, help="Image height (default: 768).")

    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)

    print(f"Loading Z-Image-Turbo pipeline...")
    pipe = LightX2VPipeline(
        model_path=os.path.join(ROOT_DIR, "models", "Z-Image-Turbo"),
        model_cls="z_image",
        task="t2i",
    )

    pipe.enable_quantize(
        dit_quantized=True,
        dit_quantized_ckpt=os.path.join(ROOT_DIR, "models", "Z-Image-Turbo-Quantized", "z_image_turbo_int8.safetensors"),
        quant_scheme="int8-torchao",
    )

    pipe.enable_offload(
        cpu_offload=True,
        offload_granularity="block",
    )

    pipe.create_generator(
        attn_mode="torch_sdpa",
        aspect_ratio=args.aspect_ratio,
        infer_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    target_shape = [int(args.height), int(args.width)] if args.width and args.height else []
    print(f"Generating image with seed {seed} at resolution {args.width}x{args.height}...")
    print(f"Prompt: {args.prompt}")

    pipe.generate(
        seed=seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        save_result_path=args.output,
        target_shape=target_shape,
    )

    if os.path.exists(args.output):
        print(f"SUCCESS: Image saved to {args.output}")
    else:
        print(f"ERROR: Image file was not created at {args.output}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
