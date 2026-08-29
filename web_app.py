#!/usr/bin/env python3
"""
The Necronomicon Vault • H. P. Lovecraft Multimodal Studio (FastAPI Backend)

A custom, cinematic, gothic Lovecraftian web UI powered by:
- FastAPI & Server-Sent Events (SSE)
- 68 Scraped Lovecraft Tales (tales/*.md)
- Ollama Lore & Synopsis (gemma4:e2b)
- Z-Image-Turbo 768x768 Cover Art (via z-image conda env)
- CosyVoice3 Vincent Price Audiobook Narration (lovecraft env)
"""

import asyncio
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

from generate_audio import clean_markdown_for_speech, get_or_create_transcript

try:
    from cosyvoice.cli.cosyvoice import CosyVoice3
    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False


app = FastAPI(title="The Necronomicon Vault", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
TALES_DIR = os.path.join(ROOT_DIR, "tales")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SYNOPSES_DIR = os.path.join(RESULTS_DIR, "synopses")
IMAGE_OUTPUT_DIR = os.path.join(ROOT_DIR, "z_image_gradio_output")
VOICE_SAMPLE = os.path.join(ROOT_DIR, "Vincent Price Voice.mp3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SYNOPSES_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


def get_saved_synopsis(tale_name: str) -> dict | None:
    """Retrieve cached synopsis data for a tale."""
    path = os.path.join(SYNOPSES_DIR, f"{tale_name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_synopsis(tale_name: str, synopsis: str, image_prompt: str = ""):
    """Persist generated or edited synopsis to disk."""
    path = os.path.join(SYNOPSES_DIR, f"{tale_name}.json")
    data = {
        "tale_name": tale_name,
        "title": tale_name.replace("_", " ").title(),
        "synopsis": synopsis,
        "image_prompt": image_prompt,
        "updated_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Mount Static & Templates
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "templates"))

# Singletons
_cosyvoice_model = None
_cached_transcript = None


def get_cosyvoice_model():
    global _cosyvoice_model
    if _cosyvoice_model is None and COSYVOICE_AVAILABLE:
        model_dir = os.path.join(ROOT_DIR, "models", "Fun-CosyVoice3-0.5B-2512")
        print(f"Loading CosyVoice3 model from {model_dir}...")
        _cosyvoice_model = CosyVoice3(model_dir, fp16=torch.cuda.is_available())
        print("CosyVoice3 loaded successfully!")
    return _cosyvoice_model


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
# API Routes
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    index_path = os.path.join(ROOT_DIR, "templates", "index.html")
    return FileResponse(index_path, media_type="text/html")


@app.get("/api/tales")
async def list_tales():
    files = sorted(glob.glob(os.path.join(TALES_DIR, "*.md")))
    tales = []
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        title = basename.replace("_", " ").title()

        # Word count approx
        with open(f, "r", encoding="utf-8") as fp:
            text = fp.read()
        words = len(text.split())

        # Check existing audio
        audio_file = os.path.join(RESULTS_DIR, f"{basename}.wav")
        has_audio = os.path.exists(audio_file)
        audio_url = f"/api/audio/{basename}.wav" if has_audio else None

        # Check existing image
        img_dir = os.path.join(IMAGE_OUTPUT_DIR, basename)
        has_image = False
        image_url = None
        if os.path.exists(img_dir):
            pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if pngs:
                has_image = True
                image_url = f"/api/image/{basename}/{os.path.basename(pngs[-1])}"

        # Check memory cover fallback
        if not has_image and basename == "memory" and os.path.exists(os.path.join(RESULTS_DIR, "memory_cover.png")):
            has_image = True
            image_url = "/api/image/memory/cover"

        # Check existing saved synopsis
        syn_data = get_saved_synopsis(basename)
        has_synopsis = syn_data is not None and bool(syn_data.get("synopsis"))
        synopsis_text = syn_data.get("synopsis") if has_synopsis else None
        image_prompt_text = syn_data.get("image_prompt") if has_synopsis else None

        tales.append({
            "name": basename,
            "title": title,
            "filename": os.path.basename(f),
            "words": words,
            "has_audio": has_audio,
            "audio_url": audio_url,
            "has_image": has_image,
            "image_url": image_url,
            "has_synopsis": has_synopsis,
            "synopsis": synopsis_text,
            "image_prompt": image_prompt_text,
        })
    return tales


@app.get("/api/tale/{tale_name}")
async def get_tale(tale_name: str):
    tale_path = os.path.join(TALES_DIR, f"{tale_name}.md")
    if not os.path.exists(tale_path):
        raise HTTPException(status_code=404, detail="Tale not found")
    with open(tale_path, "r", encoding="utf-8") as f:
        content = f.read()

    syn_data = get_saved_synopsis(tale_name)
    return {
        "name": tale_name,
        "title": tale_name.replace("_", " ").title(),
        "content": content,
        "has_synopsis": syn_data is not None and bool(syn_data.get("synopsis")),
        "synopsis": syn_data.get("synopsis") if syn_data else None,
        "image_prompt": syn_data.get("image_prompt") if syn_data else None,
    }


@app.post("/api/tale/{tale_name}/synopsis")
async def update_synopsis(tale_name: str, payload: dict):
    synopsis = payload.get("synopsis", "").strip()
    image_prompt = payload.get("image_prompt", "").strip()
    if not synopsis:
        raise HTTPException(status_code=400, detail="Synopsis cannot be empty")
    save_synopsis(tale_name, synopsis, image_prompt)
    return {"status": "ok", "message": "Synopsis saved successfully"}


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    audio_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/image/{tale_name}/{filename}")
async def serve_image(tale_name: str, filename: str):
    if filename == "cover" and tale_name == "memory":
        path = os.path.join(RESULTS_DIR, "memory_cover.png")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/png")
    image_path = os.path.join(IMAGE_OUTPUT_DIR, tale_name, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/png")


# ==========================================
# SSE Streaming Generation Endpoint
# ==========================================
@app.get("/api/generate/stream")
async def generate_stream(
    tale_name: str,
    do_summary: bool = True,
    do_image: bool = True,
    do_audio: bool = True,
    custom_prompt: str = "",
    steps: int = 8,
):
    tale_path = os.path.join(TALES_DIR, f"{tale_name}.md")
    if not os.path.exists(tale_path):
        raise HTTPException(status_code=404, detail="Tale file not found")

    with open(tale_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    tale_title = tale_name.replace("_", " ").title()

    async def event_generator():
        def sse(data: dict):
            return f"data: {json.dumps(data)}\n\n"

        yield sse({"log": f"📖 Loaded '{tale_title}' ({len(full_text):,} characters)."})
        await asyncio.sleep(0.01)

        # Step 1: Summary & Art Prompt
        image_prompt = custom_prompt.strip()
        if do_summary or (do_image and not image_prompt):
            yield sse({"log": "--- STEP 1: Consulting the Oracles of Ollama (gemma4:e2b) ---"})
            await asyncio.sleep(0.01)

            try:
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=30.0)
                prompt = (
                    f"You are an expert on H. P. Lovecraft's cosmic horror fiction.\n\n"
                    f"Here is the story '{tale_title}':\n\n{full_text[:15000]}\n\n"
                    f"Provide:\n"
                    f"1. A dark, atmospheric 3-sentence synopsis of this story.\n"
                    f"2. A detailed 1-sentence prompt for an AI image generator to create a gothic cover illustration for this story.\n\n"
                    f"Format your response as:\n"
                    f"SYNOPSIS: <synopsis>\n"
                    f"IMAGE_PROMPT: <image prompt>"
                )

                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gemma4:e2b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )

                content = resp.choices[0].message.content.strip()
                synopsis = ""
                auto_prompt = f"Gothic oil painting illustration for H. P. Lovecraft's {tale_title}, cosmic horror, dark moody lighting."
                if "SYNOPSIS:" in content and "IMAGE_PROMPT:" in content:
                    parts = content.split("IMAGE_PROMPT:")
                    synopsis = parts[0].replace("SYNOPSIS:", "").strip()
                    auto_prompt = parts[1].strip()
                else:
                    synopsis = content

                yield sse({"synopsis": synopsis, "log": f"📜 Synopsis generated ({len(synopsis)} chars)."})
                await asyncio.sleep(0.01)

                if not image_prompt:
                    image_prompt = auto_prompt
            except Exception as e:
                fallback_synopsis = full_text[:400] + "..."
                if not image_prompt:
                    image_prompt = f"Gothic oil painting illustration for '{tale_title}' by H. P. Lovecraft, dark cosmic horror, eerie moonlight."
                synopsis = f"*(Ollama notice: {e})*\n\n{fallback_synopsis}"
                yield sse({
                    "synopsis": synopsis,
                    "log": f"⚠️ Ollama notice: {e}. Used local text fallback.",
                })
                await asyncio.sleep(0.01)

            # Clean and sanitize the image prompt
            image_prompt = re.sub(r'[\r\n\t]+', ' ', image_prompt).replace('"', "'").strip()
            yield sse({"log": f"🎨 Visual Prompt: \"{image_prompt}\""})
            await asyncio.sleep(0.01)

            # Automatically persist synopsis to disk
            if synopsis:
                save_synopsis(tale_name, synopsis, image_prompt)

        # Step 2: Cover Art Generation
        if do_image and image_prompt:
            image_prompt = re.sub(r'[\r\n\t]+', ' ', image_prompt).replace('"', "'").strip()
            yield sse({"log": "--- STEP 2: Manifesting Gothic Cover Art with Z-Image-Turbo ---"})
            await asyncio.sleep(0.01)

            z_python = find_z_image_python()
            if not z_python:
                yield sse({"log": "❌ ERROR: z-image conda Python environment not found."})
                await asyncio.sleep(0.01)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_dir = os.path.join(IMAGE_OUTPUT_DIR, tale_name)
                os.makedirs(img_dir, exist_ok=True)
                out_png = os.path.join(img_dir, f"{tale_name}_{timestamp}.png")
                cli_script = os.path.join(ROOT_DIR, "generate_image.py")

                cmd = [
                    z_python, "-u",
                    cli_script,
                    "--prompt", image_prompt,
                    "--output", out_png,
                    "--aspect-ratio", "1:1",
                    "--width", "768",
                    "--height", "768",
                    "--steps", str(steps),
                ]

                yield sse({"log": f"🚀 Launching Z-Image runner (768×768, {steps} steps)..."})
                await asyncio.sleep(0.01)

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                error_lines = []
                while True:
                    line = await asyncio.to_thread(proc.stdout.readline)
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        l_clean = line.strip()
                        if l_clean:
                            error_lines.append(l_clean)
                            if "step_index" in l_clean or "infer_main" in l_clean or "SUCCESS" in l_clean:
                                yield sse({"log": f"  [Diffusion] {l_clean}"})
                                await asyncio.sleep(0.005)
                            elif "Traceback" in l_clean or "Error" in l_clean or "error:" in l_clean:
                                yield sse({"log": f"  [Error] {l_clean}"})
                                await asyncio.sleep(0.005)

                proc.stdout.close()
                await asyncio.to_thread(proc.wait)

                if proc.returncode == 0 and os.path.exists(out_png):
                    img_url = f"/api/image/{tale_name}/{os.path.basename(out_png)}"
                    yield sse({
                        "image_url": img_url,
                        "log": f"🖼️ Cover Art successfully created at 768×768!\n   Saved to: {out_png}",
                    })
                    await asyncio.sleep(0.01)
                else:
                    err_msg = "\n".join(error_lines[-5:]) if error_lines else "Unknown error"
                    yield sse({"log": f"❌ Z-Image failed (code {proc.returncode}):\n{err_msg}"})
                    await asyncio.sleep(0.01)

        # Step 3: Audiobook Narration
        if do_audio:
            yield sse({"log": "--- STEP 3: Awakening Vincent Price's Voice (CosyVoice3) ---"})
            await asyncio.sleep(0.01)

            cosyvoice = await asyncio.to_thread(get_cosyvoice_model)
            if cosyvoice is None:
                yield sse({"log": "❌ ERROR: CosyVoice model could not be initialized."})
                await asyncio.sleep(0.01)
            else:
                global _cached_transcript
                if _cached_transcript is None:
                    yield sse({"log": "Transcribing Vincent Price Voice.mp3 with Whisper..."})
                    await asyncio.sleep(0.01)
                    _cached_transcript = await asyncio.to_thread(get_or_create_transcript, VOICE_SAMPLE)
                    yield sse({"log": f"Reference transcript loaded: \"{_cached_transcript[:55]}...\""})
                    await asyncio.sleep(0.01)

                master_wav = os.path.join(RESULTS_DIR, f"{tale_name}.wav")
                if os.path.exists(master_wav):
                    yield sse({
                        "audio_url": f"/api/audio/{tale_name}.wav",
                        "log": f"🎙️ Master audiobook already exists ({master_wav}). Using cached audio.",
                    })
                    await asyncio.sleep(0.01)
                else:
                    tale_chunk_dir = os.path.join(RESULTS_DIR, tale_name)
                    os.makedirs(tale_chunk_dir, exist_ok=True)
                    chunks = clean_markdown_for_speech(full_text)
                    sample_rate = cosyvoice.sample_rate
                    audio_segments = []

                    prompt_text = _cached_transcript
                    if "<|endofprompt|>" not in prompt_text:
                        formatted_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
                    else:
                        formatted_prompt = prompt_text

                    yield sse({"log": f"Synthesizing {len(chunks)} sentence chunks (0 overlap, natural pauses)..."})
                    await asyncio.sleep(0.01)

                    for idx, chunk_info in enumerate(chunks):
                        c_text = chunk_info["text"]
                        p_ms = chunk_info["pause_ms"]
                        c_file = os.path.join(tale_chunk_dir, f"part_{idx:04d}.wav")

                        t0 = time.time()
                        if os.path.exists(c_file) and os.path.getsize(c_file) > 1000:
                            a_data, _ = sf.read(c_file, dtype="float32")
                            yield sse({"log": f"  [Chunk {idx+1}/{len(chunks)}] Loaded cached part_{idx:04d}.wav"})
                            await asyncio.sleep(0.005)
                        else:
                            try:
                                yield sse({"log": f"  [Chunk {idx+1}/{len(chunks)}] Narrating: \"{c_text[:50]}...\""})
                                await asyncio.sleep(0.005)

                                output = await asyncio.to_thread(
                                    cosyvoice.inference_zero_shot,
                                    c_text, formatted_prompt, VOICE_SAMPLE
                                )
                                t_list = [c["tts_speech"] for c in output]
                                if not t_list:
                                    continue
                                t_full = torch.cat(t_list, dim=1).squeeze().cpu().numpy()
                                sf.write(c_file, t_full, sample_rate)
                                a_data = t_full
                                dt = time.time() - t0
                                dur = len(a_data) / sample_rate
                                yield sse({"log": f"  -> Chunk {idx+1} synthesized ({dur:.1f}s audio in {dt:.2f}s)"})
                                await asyncio.sleep(0.005)
                            except Exception as ce:
                                yield sse({"log": f"  [ERROR] Chunk {idx} failed: {ce}"})
                                await asyncio.sleep(0.005)
                                continue

                        audio_segments.append(a_data)
                        if p_ms > 0:
                            audio_segments.append(np.zeros(int(sample_rate * (p_ms / 1000.0)), dtype=np.float32))

                    if audio_segments:
                        master_audio = np.concatenate(audio_segments)
                        sf.write(master_wav, master_audio, sample_rate)
                        tot_sec = len(master_audio) / sample_rate
                        mins = int(tot_sec // 60)
                        secs = int(tot_sec % 60)
                        yield sse({
                            "audio_url": f"/api/audio/{tale_name}.wav",
                            "log": f"🎙️ Master audiobook completed! ({mins}m {secs}s) -> {master_wav}",
                        })
                        await asyncio.sleep(0.01)

        yield sse({"done": True, "log": "=== 🌟 Necronomicon Vault synthesis complete! ==="})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🌌 THE NECRONOMICON VAULT • H. P. Lovecraft Multimodal Studio")
    print("Launching on: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
