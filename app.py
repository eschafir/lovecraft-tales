#!/usr/bin/env python3
"""
Lovecraft Multimodal Studio (Gradio App)

Integrates:
1. 68 Scraped Lovecraft Tales (tales/*.md)
2. Story Synopsis & Image Prompt Generation (Ollama: gemma4:e2b)
3. Gothic Cover Art Generation (Z-Image-Turbo / LightX2V via z-image env)
4. Audiobook Narration (CosyVoice3 + Vincent Price voice cloning)
5. Real-Time Streaming Console Output & Logs
"""

import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime
import gradio as gr
import numpy as np
import soundfile as sf
import torch
from openai import OpenAI

# Fix PATH for FFmpeg / torchcodec DLLs on Windows
env_dir = os.path.dirname(sys.executable)
lib_bin = os.path.join(env_dir, "Library", "bin")
if os.path.exists(lib_bin) and lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")

# CosyVoice import path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_DIR = os.path.join(ROOT_DIR, "CosyVoice")
sys.path.append(COSYVOICE_DIR)
sys.path.append(os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS"))

from generate_audio import (
    clean_markdown_for_speech,
    get_or_create_transcript,
)

try:
    from cosyvoice.cli.cosyvoice import CosyVoice3
    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False


# ==========================================
# Lazy-loaded Model Singletons
# ==========================================
_cosyvoice_model = None
_cached_transcript = None

TALES_DIR = os.path.join(ROOT_DIR, "tales")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
IMAGE_OUTPUT_DIR = os.path.join(ROOT_DIR, "z_image_gradio_output")
VOICE_SAMPLE = os.path.join(ROOT_DIR, "Vincent Price Voice.mp3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


def get_cosyvoice_model(log_callback=print):
    global _cosyvoice_model
    if _cosyvoice_model is None and COSYVOICE_AVAILABLE:
        model_dir = os.path.join(ROOT_DIR, "models", "Fun-CosyVoice3-0.5B-2512")
        log_callback(f"Loading CosyVoice3 model from {model_dir} (fp16={torch.cuda.is_available()})...")
        _cosyvoice_model = CosyVoice3(model_dir, fp16=torch.cuda.is_available())
        log_callback("CosyVoice3 loaded successfully!")
    return _cosyvoice_model


# ==========================================
# Tale Discovery & Loading
# ==========================================
def get_available_tales() -> list[tuple[str, str]]:
    """Returns list of (display_title, filepath)."""
    files = sorted(glob.glob(os.path.join(TALES_DIR, "*.md")))
    tales = []
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        display_title = basename.replace("_", " ").title()
        tales.append((display_title, f))
    return tales


ALL_TALES = get_available_tales()
TALE_DICT = {title: path for title, path in ALL_TALES}


def find_z_image_python() -> str:
    """Locate the Python executable for the z-image conda environment."""
    candidates = [
        r"C:\Users\esteb\miniconda3\envs\z-image\python.exe",
        os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "z-image", "python.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ==========================================
# Streaming Generation Helpers
# ==========================================
def generate_synopsis_and_art_prompt(tale_text: str, tale_title: str, log_callback=print) -> tuple[str, str]:
    """Call Ollama gemma4:e2b to generate a summary and image generation prompt."""
    try:
        log_callback("Connecting to Ollama at http://localhost:11434/v1 (model: gemma4:e2b)...")
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=30.0)
        truncated_text = tale_text[:15000]
        prompt = (
            f"You are an expert on H. P. Lovecraft's cosmic horror fiction.\n\n"
            f"Here is the story '{tale_title}':\n\n{truncated_text}\n\n"
            f"Provide:\n"
            f"1. A dark, atmospheric 3-sentence synopsis of this story.\n"
            f"2. A detailed 1-sentence prompt for an AI image generator to create a gothic cover illustration for this story.\n\n"
            f"Format your response as:\n"
            f"SYNOPSIS: <synopsis>\n"
            f"IMAGE_PROMPT: <image prompt>"
        )
        resp = client.chat.completions.create(
            model="gemma4:e2b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()

        synopsis = ""
        image_prompt = f"Gothic oil painting illustration for H. P. Lovecraft's {tale_title}, cosmic horror, dark moody lighting, highly detailed masterpiece."

        if "SYNOPSIS:" in content and "IMAGE_PROMPT:" in content:
            parts = content.split("IMAGE_PROMPT:")
            synopsis = parts[0].replace("SYNOPSIS:", "").strip()
            image_prompt = parts[1].strip()
        else:
            synopsis = content

        log_callback(f"Synopsis received ({len(synopsis)} chars).")
        log_callback(f"Crafted Image Prompt: {image_prompt}")
        return synopsis, image_prompt
    except Exception as e:
        log_callback(f"Ollama connection notice: {e}")
        log_callback("Using local text excerpt for synopsis.")
        fallback_summary = f"*(Ollama unavailable - text excerpt)*\n\n" + tale_text[:400] + "..."
        fallback_prompt = f"Gothic cosmic horror oil painting illustration for '{tale_title}' by H. P. Lovecraft, dark eerie atmosphere."
        return fallback_summary, fallback_prompt


def stream_cover_image(prompt: str, tale_name: str, steps: int = 8):
    """
    Generator that runs Z-Image-Turbo in the z-image environment,
    yielding console lines in real time and finally yielding the saved image path.
    """
    z_python = find_z_image_python()
    if not z_python:
        yield "[WARN] z-image Python environment not found.\n", None
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(IMAGE_OUTPUT_DIR, tale_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tale_name}_{timestamp}.png")
    cli_script = os.path.join(ROOT_DIR, "generate_image.py")

    cmd = [
        z_python,
        "-u",  # Unbuffered stdout for immediate streaming
        cli_script,
        "--prompt", prompt,
        "--output", out_path,
        "--aspect-ratio", "1:1",
        "--width", "768",
        "--height", "768",
        "--steps", str(steps),
    ]

    yield f"🚀 Launching Z-Image-Turbo subprocess via '{z_python}'...\n", None
    yield f"🎨 Prompt: \"{prompt}\" (Resolution: 768×768, Steps: {steps})\n", None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in iter(proc.stdout.readline, ""):
        if not line and proc.poll() is not None:
            break
        if line:
            # Filter noise and yield meaningful progress
            cleaned = line.strip()
            if cleaned:
                yield f"[Z-Image] {cleaned}\n", None

    proc.stdout.close()
    proc.wait()

    if proc.returncode == 0 and os.path.exists(out_path):
        yield f"\n✅ Cover art successfully generated and saved to:\n   {out_path}\n", out_path
    else:
        yield f"\n❌ Z-Image generation process exited with code {proc.returncode}\n", None


def stream_tale_audiobook(tale_path: str, log_callback=print):
    """
    Synthesizes the tale into a master WAV file with real-time chunk logging.
    Yields (log_message, final_audio_path_or_None).
    """
    global _cached_transcript
    cosyvoice = get_cosyvoice_model(log_callback=log_callback)
    if cosyvoice is None:
        yield "[ERROR] CosyVoice model could not be initialized.\n", None
        return

    if _cached_transcript is None:
        yield "Transcribing reference audio (Vincent Price Voice.mp3) with Whisper...\n", None
        _cached_transcript = get_or_create_transcript(VOICE_SAMPLE)
        yield f"Reference transcript cached: \"{_cached_transcript[:60]}...\"\n", None

    tale_name = os.path.splitext(os.path.basename(tale_path))[0]
    master_file = os.path.join(RESULTS_DIR, f"{tale_name}.wav")

    # Resumption: if master already exists
    if os.path.exists(master_file):
        yield f"✅ Master audiobook already exists: {master_file} (skipping synthesis)\n", master_file
        return

    tale_chunk_dir = os.path.join(RESULTS_DIR, tale_name)
    os.makedirs(tale_chunk_dir, exist_ok=True)

    with open(tale_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    chunks = clean_markdown_for_speech(md_content)
    if not chunks:
        yield f"[WARN] No readable chunks found in {tale_path}\n", None
        return

    sample_rate = cosyvoice.sample_rate
    audio_segments = []

    prompt_text = _cached_transcript
    if "<|endofprompt|>" not in prompt_text:
        formatted_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
    else:
        formatted_prompt = prompt_text

    yield f"🎙️ Narrating '{tale_name}' ({len(chunks)} audio chunks)...\n", None

    for idx, chunk_info in enumerate(chunks):
        chunk_text = chunk_info["text"]
        pause_ms = chunk_info["pause_ms"]
        chunk_file = os.path.join(tale_chunk_dir, f"part_{idx:04d}.wav")

        t0 = time.time()
        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1000:
            audio_data, _ = sf.read(chunk_file, dtype="float32")
            yield f"   [Chunk {idx+1}/{len(chunks)}] Loaded cached part_{idx:04d}.wav\n", None
        else:
            try:
                yield f"   [Chunk {idx+1}/{len(chunks)}] Synthesizing: \"{chunk_text[:55]}...\"\n", None
                output = cosyvoice.inference_zero_shot(chunk_text, formatted_prompt, VOICE_SAMPLE)
                audio_tensors = [c["tts_speech"] for c in output]
                if not audio_tensors:
                    continue
                audio_tensor = torch.cat(audio_tensors, dim=1).squeeze().cpu().numpy()
                sf.write(chunk_file, audio_tensor, sample_rate)
                audio_data = audio_tensor
                dur = len(audio_data) / sample_rate
                t_cost = time.time() - t0
                yield f"   -> Generated {dur:.1f}s speech in {t_cost:.2f}s\n", None
            except Exception as e:
                yield f"   [ERROR] Failed chunk {idx}: {e}\n", None
                continue

        audio_segments.append(audio_data)

        if pause_ms > 0:
            pause_samples = int(sample_rate * (pause_ms / 1000.0))
            audio_segments.append(np.zeros(pause_samples, dtype=np.float32))

    if not audio_segments:
        yield f"❌ No audio segments generated for {tale_name}\n", None
        return

    # Stitch master audiobook
    master_audio = np.concatenate(audio_segments)
    sf.write(master_file, master_audio, sample_rate)
    total_seconds = len(master_audio) / sample_rate
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    yield f"\n✅ Master Audiobook ready! Duration: {minutes}m {seconds}s\n   Saved to: {master_file}\n", master_file


# ==========================================
# Main Studio Orchestration (Generator)
# ==========================================
def process_tale_streaming(
    selected_title: str,
    do_summary: bool,
    do_image: bool,
    do_audio: bool,
    custom_prompt: str,
    image_steps: int,
):
    """
    Main generator function for Gradio that streams console updates live.
    Yields (synopsis_md, cover_image_path, audio_path, live_console_text).
    """
    if not selected_title or selected_title not in TALE_DICT:
        yield "*Please select a valid story.*", None, None, "ERROR: No story selected."
        return

    tale_path = TALE_DICT[selected_title]
    tale_name = os.path.splitext(os.path.basename(tale_path))[0]

    with open(tale_path, "r", encoding="utf-8") as f:
        full_story_text = f.read()

    synopsis_out = "*Generating...*"
    image_out = None
    audio_out = None
    log_lines = []

    def append_log(msg: str):
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg.rstrip()}")
        return "\n".join(log_lines)

    log_text = append_log(f"Selected Tale: {selected_title} ({os.path.basename(tale_path)})")
    yield synopsis_out, image_out, audio_out, log_text

    # 1. Summary & Prompt
    image_prompt = custom_prompt.strip()
    if do_summary or (do_image and not image_prompt):
        log_text = append_log("--- STEP 1: Story Synopsis & Art Prompt (Ollama) ---")
        yield synopsis_out, image_out, audio_out, log_text

        def log_cb(msg):
            nonlocal log_text
            log_text = append_log(f"  {msg}")

        synopsis, auto_prompt = generate_synopsis_and_art_prompt(full_story_text, selected_title, log_callback=log_cb)
        synopsis_out = f"### 📜 Synopsis\n\n{synopsis}"
        if not image_prompt:
            image_prompt = auto_prompt

        yield synopsis_out, image_out, audio_out, log_text

    # 2. Cover Art
    if do_image:
        log_text = append_log("--- STEP 2: Gothic Cover Art Generation (Z-Image-Turbo) ---")
        yield synopsis_out, image_out, audio_out, log_text

        for log_msg, img_path in stream_cover_image(image_prompt, tale_name, steps=int(image_steps)):
            if log_msg:
                log_lines.append(log_msg.rstrip())
                log_text = "\n".join(log_lines)
            if img_path:
                image_out = img_path
            yield synopsis_out, image_out, audio_out, log_text

    # 3. Audiobook Narration
    if do_audio:
        log_text = append_log("--- STEP 3: Vincent Price Audiobook Narration (CosyVoice3) ---")
        yield synopsis_out, image_out, audio_out, log_text

        def log_cb(msg):
            nonlocal log_text
            log_text = append_log(f"  {msg}")

        for log_msg, aud_path in stream_tale_audiobook(tale_path, log_callback=log_cb):
            if log_msg:
                log_lines.append(log_msg.rstrip())
                log_text = "\n".join(log_lines)
            if aud_path:
                audio_out = aud_path
            yield synopsis_out, image_out, audio_out, log_text

    log_text = append_log("=== 🎉 All requested steps completed! ===")
    yield synopsis_out, image_out, audio_out, log_text


def load_story_preview(selected_title: str) -> str:
    """Display the text preview when a story is chosen from the dropdown."""
    if not selected_title or selected_title not in TALE_DICT:
        return ""
    tale_path = TALE_DICT[selected_title]
    with open(tale_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:2500] + ("\n\n... [Full story loaded for generation] ..." if len(text) > 2500 else "")


# ==========================================
# Gradio UI Layout
# ==========================================
tale_titles = [title for title, _ in ALL_TALES]
default_tale = "Dagon" if "Dagon" in tale_titles else (tale_titles[0] if tale_titles else None)

custom_css = """
body { background-color: #0b0c10; color: #c5c6c7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.gradio-container { max-width: 1280px !important; }
h1, h2, h3 { color: #66fcf1 !important; }
.console-box textarea {
    font-family: 'Cascadia Code', Consolas, 'Courier New', monospace !important;
    background-color: #050608 !important;
    color: #45f3ff !important;
    font-size: 13px !important;
    line-height: 1.4 !important;
}
"""

with gr.Blocks(title="Lovecraft Multimodal Studio") as demo:
    gr.Markdown("# 🌌 H. P. Lovecraft Multimodal Studio")
    gr.Markdown(
        "*Select any of the 68 Lovecraft fiction tales to generate an AI synopsis, gothic cover illustration, and cloned Vincent Price audio narration.*"
    )

    with gr.Row():
        # Left Column: Inputs & Controls
        with gr.Column(scale=1):
            tale_dropdown = gr.Dropdown(
                choices=tale_titles,
                value=default_tale,
                label="📖 Select Lovecraft Tale",
                interactive=True,
            )

            with gr.Accordion("🔍 Story Text Preview", open=False):
                story_preview = gr.Textbox(
                    value=load_story_preview(default_tale) if default_tale else "",
                    lines=8,
                    label="Original Text Excerpt",
                    interactive=False,
                )

            gr.Markdown("### ⚙️ Pipeline Configuration")
            chk_summary = gr.Checkbox(value=True, label="1. Story Synopsis (Ollama: gemma4:e2b)")
            chk_image = gr.Checkbox(value=True, label="2. Gothic Cover Art (Z-Image-Turbo)")
            chk_audio = gr.Checkbox(value=True, label="3. Audiobook Narration (CosyVoice - Vincent Price)")

            with gr.Row():
                image_steps_slider = gr.Slider(
                    minimum=4,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Z-Image Diffusion Steps",
                    info="Fewer steps = faster generation on GPU",
                )

            custom_prompt_input = gr.Textbox(
                label="🎨 Custom Cover Art Prompt (Optional)",
                placeholder="Leave blank to auto-generate from story synopsis...",
                lines=2,
            )

            btn_generate = gr.Button("⚡ Generate Tale Experience", variant="primary", size="lg")

        # Right Column: Outputs & Live Console
        with gr.Column(scale=1):
            synopsis_box = gr.Markdown(value="*Select a tale and click Generate.*")

            cover_image = gr.Image(label="🖼️ Gothic Cover Illustration", type="filepath")

            audio_player = gr.Audio(label="🎙️ Vincent Price Narration", type="filepath")

    # Bottom Full-Width Section: Real-Time Terminal Console
    with gr.Row():
        with gr.Column(scale=1):
            console_box = gr.Textbox(
                label="📟 Live Console Logs & Execution Output",
                lines=12,
                max_lines=25,
                autoscroll=True,
                interactive=False,
                elem_classes=["console-box"],
                value="[System Ready] Select a tale and click 'Generate Tale Experience' to see real-time execution logs.\n",
            )

    # Event bindings
    tale_dropdown.change(fn=load_story_preview, inputs=[tale_dropdown], outputs=[story_preview])

    btn_generate.click(
        fn=process_tale_streaming,
        inputs=[
            tale_dropdown,
            chk_summary,
            chk_image,
            chk_audio,
            custom_prompt_input,
            image_steps_slider,
        ],
        outputs=[synopsis_box, cover_image, audio_player, console_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)
