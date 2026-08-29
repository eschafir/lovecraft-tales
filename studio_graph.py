#!/usr/bin/env python3
"""
The Necronomicon Vault • LangGraph Multimodal Workflow Engine

Orchestrates:
1. Load Story Node: Document retrieval & state initialization
2. Generate Lore Node: Ollama (gemma4:e2b) synopsis & visual prompt engineering
3. Generate Cover Node: Z-Image-Turbo 768x768 int8 diffusion via z-image env
4. Generate Audio Node: CosyVoice3 Vincent Price zero-shot voice cloning
5. Conditional Routing & MemorySaver Checkpointing
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import soundfile as sf
import torch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from openai import OpenAI

# Fix PATH for DLLs on Windows
env_dir = os.path.dirname(sys.executable)
lib_bin = os.path.join(env_dir, "Library", "bin")
if os.path.exists(lib_bin) and lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")

# CosyVoice imports
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


# Constants & Paths
TALES_DIR = os.path.join(ROOT_DIR, "tales")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SYNOPSES_DIR = os.path.join(RESULTS_DIR, "synopses")
IMAGE_OUTPUT_DIR = os.path.join(ROOT_DIR, "z_image_gradio_output")
VOICE_SAMPLE = os.path.join(ROOT_DIR, "Vincent Price Voice.mp3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SYNOPSES_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

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
    candidates = [
        r"C:\Users\esteb\miniconda3\envs\z-image\python.exe",
        os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "z-image", "python.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_saved_synopsis(tale_name: str) -> dict | None:
    path = os.path.join(SYNOPSES_DIR, f"{tale_name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_synopsis(tale_name: str, synopsis: str, image_prompt: str = ""):
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


# ==========================================
# LangGraph State Schema
# ==========================================
class StudioState(TypedDict):
    tale_name: str
    tale_title: str
    full_text: str
    
    # Configuration inputs
    do_summary: bool
    do_image: bool
    do_audio: bool
    custom_prompt: str
    image_steps: int
    
    # Generated Outputs
    synopsis: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    image_url: Optional[str]
    audio_path: Optional[str]
    audio_url: Optional[str]
    
    # Live streaming status & logs
    logs: List[str]
    current_node: str
    error: Optional[str]


# ==========================================
# LangGraph Nodes with Live Callbacks
# ==========================================
def load_story_node(state: StudioState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Node 1: Load Markdown text and check existing caches."""
    cfg = config.get("configurable", {}) if config else {}
    log_cb = cfg.get("log_cb")
    
    tale_name = state["tale_name"]
    tale_path = os.path.join(TALES_DIR, f"{tale_name}.md")
    
    if not os.path.exists(tale_path):
        err_msg = f"❌ Error: Story file not found: {tale_path}"
        if log_cb:
            log_cb(err_msg)
        return {
            "error": err_msg,
            "logs": state.get("logs", []) + [err_msg],
            "current_node": "load_story",
        }
    
    with open(tale_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    tale_title = tale_name.replace("_", " ").title()
    syn_data = get_saved_synopsis(tale_name)
    existing_synopsis = syn_data.get("synopsis") if syn_data else state.get("synopsis")
    existing_prompt = syn_data.get("image_prompt") if syn_data else state.get("image_prompt")
    
    log_msg = f"📖 [LangGraph: LoadStory] Loaded '{tale_title}' ({len(full_text):,} chars)."
    if log_cb:
        log_cb(log_msg)
        
    return {
        "tale_title": tale_title,
        "full_text": full_text,
        "synopsis": existing_synopsis,
        "image_prompt": existing_prompt,
        "logs": state.get("logs", []) + [log_msg],
        "current_node": "load_story",
    }


def generate_lore_node(state: StudioState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Node 2: Consult Ollama gemma4:e2b for atmospheric synopsis & visual prompt."""
    cfg = config.get("configurable", {}) if config else {}
    log_cb = cfg.get("log_cb")
    synopsis_cb = cfg.get("synopsis_cb")
    
    tale_name = state["tale_name"]
    tale_title = state["tale_title"]
    full_text = state["full_text"]
    custom_prompt = state.get("custom_prompt", "").strip()
    logs = list(state.get("logs", []))
    
    synopsis = state.get("synopsis")
    image_prompt = custom_prompt or state.get("image_prompt")
    
    # If summary is requested or visual prompt is needed
    if state["do_summary"] or (state["do_image"] and not image_prompt):
        msg1 = "📜 [LangGraph: LoreEngine] Consulting Ollama (gemma4:e2b)..."
        logs.append(msg1)
        if log_cb:
            log_cb(msg1)
            
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
            resp = client.chat.completions.create(
                model="gemma4:e2b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = resp.choices[0].message.content.strip()
            
            auto_prompt = f"Gothic oil painting illustration for H. P. Lovecraft's {tale_title}, cosmic horror, dark moody lighting."
            if "SYNOPSIS:" in content and "IMAGE_PROMPT:" in content:
                parts = content.split("IMAGE_PROMPT:")
                synopsis = parts[0].replace("SYNOPSIS:", "").strip()
                auto_prompt = parts[1].strip()
            else:
                synopsis = content
                
            if not custom_prompt:
                image_prompt = auto_prompt
                
            msg2 = f"📜 Synopsis created ({len(synopsis)} chars)."
            logs.append(msg2)
            if log_cb:
                log_cb(msg2)
            if synopsis_cb:
                synopsis_cb(synopsis)
        except Exception as e:
            fallback = full_text[:400] + "..."
            synopsis = f"*(Ollama note: {e})*\n\n{fallback}"
            if not image_prompt:
                image_prompt = f"Gothic oil painting illustration for '{tale_title}' by H. P. Lovecraft, dark cosmic horror, eerie moonlight."
            msg_err = f"⚠️ Ollama note: {e}. Used excerpt."
            logs.append(msg_err)
            if log_cb:
                log_cb(msg_err)
            if synopsis_cb:
                synopsis_cb(synopsis)
            
        if image_prompt:
            image_prompt = re.sub(r'[\r\n\t]+', ' ', image_prompt).replace('"', "'").strip()
            msg_p = f"🎨 Visual Prompt: \"{image_prompt}\""
            logs.append(msg_p)
            if log_cb:
                log_cb(msg_p)
            
        if synopsis:
            save_synopsis(tale_name, synopsis, image_prompt or "")

    return {
        "synopsis": synopsis,
        "image_prompt": image_prompt,
        "logs": logs,
        "current_node": "generate_lore",
    }


def generate_cover_node(state: StudioState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Node 3: Z-Image-Turbo (int8) diffusion runner via z-image conda env."""
    if not state.get("do_image"):
        return {"current_node": "generate_cover"}
        
    cfg = config.get("configurable", {}) if config else {}
    log_cb = cfg.get("log_cb")
    image_cb = cfg.get("image_cb")
    
    tale_name = state["tale_name"]
    image_prompt = state.get("image_prompt") or f"Gothic illustration for {state['tale_title']}"
    image_prompt = re.sub(r'[\r\n\t]+', ' ', image_prompt).replace('"', "'").strip()
    steps = state.get("image_steps", 8)
    logs = list(state.get("logs", []))
    
    z_python = find_z_image_python()
    if not z_python:
        err = "❌ ERROR: z-image conda Python environment not found."
        logs.append(err)
        if log_cb:
            log_cb(err)
        return {"logs": logs, "error": "z-image Python missing", "current_node": "generate_cover"}
        
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
    
    launch_msg = f"🖼️ [LangGraph: Diffusion] Launching Z-Image runner (768×768, {steps} steps)..."
    logs.append(launch_msg)
    if log_cb:
        log_cb(launch_msg)
        
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
            l_clean = line.strip()
            if "step_index" in l_clean or "infer_main" in l_clean or "SUCCESS" in l_clean:
                line_msg = f"  [Diffusion] {l_clean}"
                logs.append(line_msg)
                if log_cb:
                    log_cb(line_msg)
                
    proc.stdout.close()
    proc.wait()
    
    image_url = None
    if proc.returncode == 0 and os.path.exists(out_png):
        image_url = f"/api/image/{tale_name}/{os.path.basename(out_png)}"
        done_msg = f"🖼️ Cover Art created at 768×768! -> {out_png}"
        logs.append(done_msg)
        if log_cb:
            log_cb(done_msg)
        if image_cb:
            image_cb(image_url)
    else:
        fail_msg = f"❌ Diffusion process exited with code {proc.returncode}"
        logs.append(fail_msg)
        if log_cb:
            log_cb(fail_msg)
        
    return {
        "image_path": out_png if image_url else None,
        "image_url": image_url,
        "logs": logs,
        "current_node": "generate_cover",
    }


def generate_audio_node(state: StudioState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Node 4: CosyVoice3 TTS with Vincent Price zero-shot voice cloning."""
    if not state.get("do_audio"):
        return {"current_node": "generate_audio"}
        
    cfg = config.get("configurable", {}) if config else {}
    log_cb = cfg.get("log_cb")
    audio_cb = cfg.get("audio_cb")
    
    tale_name = state["tale_name"]
    full_text = state["full_text"]
    logs = list(state.get("logs", []))
    
    cosyvoice = get_cosyvoice_model()
    if cosyvoice is None:
        err = "❌ ERROR: CosyVoice model could not be initialized."
        logs.append(err)
        if log_cb:
            log_cb(err)
        return {"logs": logs, "error": "CosyVoice unavailable", "current_node": "generate_audio"}
        
    global _cached_transcript
    if _cached_transcript is None:
        tr_msg = "Transcribing Vincent Price Voice.mp3 with Whisper..."
        logs.append(tr_msg)
        if log_cb:
            log_cb(tr_msg)
        _cached_transcript = get_or_create_transcript(VOICE_SAMPLE)
        cached_tr_msg = f"Reference transcript cached: \"{_cached_transcript[:55]}...\""
        logs.append(cached_tr_msg)
        if log_cb:
            log_cb(cached_tr_msg)
        
    master_wav = os.path.join(RESULTS_DIR, f"{tale_name}.wav")
    if os.path.exists(master_wav):
        audio_url = f"/api/audio/{tale_name}.wav"
        exists_msg = f"🎙️ Master audiobook already exists ({master_wav}). Loaded from cache."
        logs.append(exists_msg)
        if log_cb:
            log_cb(exists_msg)
        if audio_cb:
            audio_cb(audio_url)
        return {
            "audio_path": master_wav,
            "audio_url": audio_url,
            "logs": logs,
            "current_node": "generate_audio",
        }
        
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
        
    start_synth_msg = f"🎙️ [LangGraph: AudioEngine] Synthesizing {len(chunks)} sentence chunks (Vincent Price voice)..."
    logs.append(start_synth_msg)
    if log_cb:
        log_cb(start_synth_msg)
    
    for idx, chunk_info in enumerate(chunks):
        c_text = chunk_info["text"]
        p_ms = chunk_info["pause_ms"]
        c_file = os.path.join(tale_chunk_dir, f"part_{idx:04d}.wav")
        
        t0 = time.time()
        if os.path.exists(c_file) and os.path.getsize(c_file) > 1000:
            a_data, _ = sf.read(c_file, dtype="float32")
            ch_msg = f"  [Chunk {idx+1}/{len(chunks)}] Loaded cached part_{idx:04d}.wav"
            logs.append(ch_msg)
            if log_cb:
                log_cb(ch_msg)
        else:
            try:
                narr_msg = f"  [Chunk {idx+1}/{len(chunks)}] Narrating: \"{c_text[:50]}...\""
                logs.append(narr_msg)
                if log_cb:
                    log_cb(narr_msg)
                output = cosyvoice.inference_zero_shot(c_text, formatted_prompt, VOICE_SAMPLE)
                t_list = [c["tts_speech"] for c in output]
                if not t_list:
                    continue
                t_full = torch.cat(t_list, dim=1).squeeze().cpu().numpy()
                sf.write(c_file, t_full, sample_rate)
                a_data = t_full
                dt = time.time() - t0
                dur = len(a_data) / sample_rate
                done_ch_msg = f"  -> Chunk {idx+1} synthesized ({dur:.1f}s audio in {dt:.2f}s)"
                logs.append(done_ch_msg)
                if log_cb:
                    log_cb(done_ch_msg)
            except Exception as ce:
                err_ch = f"  [ERROR] Chunk {idx} failed: {ce}"
                logs.append(err_ch)
                if log_cb:
                    log_cb(err_ch)
                continue
                
        audio_segments.append(a_data)
        if p_ms > 0:
            audio_segments.append(np.zeros(int(sample_rate * (p_ms / 1000.0)), dtype=np.float32))
            
    audio_url = None
    if audio_segments:
        master_audio = np.concatenate(audio_segments)
        sf.write(master_wav, master_audio, sample_rate)
        tot_sec = len(master_audio) / sample_rate
        mins = int(tot_sec // 60)
        secs = int(tot_sec % 60)
        audio_url = f"/api/audio/{tale_name}.wav"
        fin_msg = f"🎙️ Master audiobook completed! ({mins}m {secs}s) -> {master_wav}"
        logs.append(fin_msg)
        if log_cb:
            log_cb(fin_msg)
        if audio_cb:
            audio_cb(audio_url)
        
    return {
        "audio_path": master_wav if audio_url else None,
        "audio_url": audio_url,
        "logs": logs,
        "current_node": "generate_audio",
    }


# ==========================================
# Conditional Routing Logic
# ==========================================
def route_after_lore(state: StudioState) -> str:
    if state.get("do_image"):
        return "generate_cover"
    elif state.get("do_audio"):
        return "generate_audio"
    return END


def route_after_cover(state: StudioState) -> str:
    if state.get("do_audio"):
        return "generate_audio"
    return END


# ==========================================
# LangGraph Graph Construction
# ==========================================
def build_studio_graph():
    builder = StateGraph(StudioState)
    
    builder.add_node("load_story", load_story_node)
    builder.add_node("generate_lore", generate_lore_node)
    builder.add_node("generate_cover", generate_cover_node)
    builder.add_node("generate_audio", generate_audio_node)
    
    builder.add_edge(START, "load_story")
    builder.add_edge("load_story", "generate_lore")
    
    # Conditional branching with explicit edge mappings for visualization
    builder.add_conditional_edges(
        "generate_lore",
        route_after_lore,
        {
            "generate_cover": "generate_cover",
            "generate_audio": "generate_audio",
            END: END,
        }
    )
    builder.add_conditional_edges(
        "generate_cover",
        route_after_cover,
        {
            "generate_audio": "generate_audio",
            END: END,
        }
    )
    builder.add_edge("generate_audio", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


studio_graph = build_studio_graph()


# Direct CLI Test
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("Testing LangGraph Tale Studio Engine on 'memory'...")
    initial_input: StudioState = {
        "tale_name": "memory",
        "tale_title": "",
        "full_text": "",
        "do_summary": True,
        "do_image": False,
        "do_audio": False,
        "custom_prompt": "",
        "image_steps": 4,
        "synopsis": None,
        "image_prompt": None,
        "image_path": None,
        "image_url": None,
        "audio_path": None,
        "audio_url": None,
        "logs": [],
        "current_node": "",
        "error": None,
    }
    
    config = {"configurable": {"thread_id": "cli_test_memory"}}
    for output in studio_graph.stream(initial_input, config=config):
        for node_name, node_state in output.items():
            print(f"\n--- [Completed Node: {node_name}] ---")
            if "logs" in node_state:
                for l in node_state["logs"][-2:]:
                    print("  >", l)
