#!/usr/bin/env python3
"""
The Necronomicon Vault • H. P. Lovecraft Multimodal Studio (FastAPI + LangGraph)

Stateful, Graph-based Orchestration of:
- LangGraph Workflow Engine (studio_graph.py)
- 68 Scraped Lovecraft Tales (tales/*.md)
- Ollama Lore & Synopsis (gemma4:e2b)
- Z-Image-Turbo 768x768 Cover Art (via z-image conda env)
- CosyVoice3 Vincent Price Audiobook Narration
- Real-Time Server-Sent Events (SSE) Execution Streaming
"""

import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from studio_graph import (
    IMAGE_OUTPUT_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    SYNOPSES_DIR,
    TALES_DIR,
    StudioState,
    get_saved_synopsis,
    save_synopsis,
    studio_graph,
)

app = FastAPI(title="The Necronomicon Vault (LangGraph)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "static")), name="static")


# ==========================================
# Frontend & Catalog Routes
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


@app.get("/api/graph/dag")
async def serve_dag():
    dag_path = os.path.join(RESULTS_DIR, "langgraph_dag.png")
    if os.path.exists(dag_path):
        return FileResponse(dag_path, media_type="image/png")
    try:
        png_bytes = studio_graph.get_graph().draw_mermaid_png()
        with open(dag_path, "wb") as f:
            f.write(png_bytes)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not render DAG: {e}")


# ==========================================
# LangGraph Streaming SSE Endpoint
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

    initial_state: StudioState = {
        "tale_name": tale_name,
        "tale_title": tale_name.replace("_", " ").title(),
        "full_text": "",
        "do_summary": do_summary,
        "do_image": do_image,
        "do_audio": do_audio,
        "custom_prompt": custom_prompt,
        "image_steps": steps,
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

    async def event_generator():
        def sse(data: dict):
            return f"data: {json.dumps(data)}\n\n"

        yield sse({"log": f"🔮 [LangGraph] Initialized graph execution for '{initial_state['tale_title']}'."})
        await asyncio.sleep(0.01)

        queue = asyncio.Queue()

        def log_cb(msg: str):
            queue.put_nowait({"log": msg})

        def synopsis_cb(text: str):
            queue.put_nowait({"synopsis": text})

        def image_cb(url: str):
            queue.put_nowait({"image_url": url})

        def audio_cb(url: str):
            queue.put_nowait({"audio_url": url})

        thread_id = f"session_{tale_name}_{int(time.time())}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "log_cb": log_cb,
                "synopsis_cb": synopsis_cb,
                "image_cb": image_cb,
                "audio_cb": audio_cb,
            }
        }

        # Run LangGraph execution asynchronously in background task
        graph_task = asyncio.create_task(studio_graph.ainvoke(initial_state, config=config))

        # Stream queue events in real-time as they are produced by nodes
        while not graph_task.done() or not queue.empty():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield sse(item)
                await asyncio.sleep(0.001)
            except asyncio.TimeoutError:
                continue

        try:
            await graph_task
            yield sse({"done": True, "log": "=== 🌟 [LangGraph] Multimodal Studio synthesis complete! ==="})
        except Exception as e:
            yield sse({"done": True, "log": f"❌ [LangGraph Error] {str(e)}"})

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("🌌 THE NECRONOMICON VAULT • H. P. Lovecraft Multimodal Studio")
    print("Orchestrator: LangGraph State Machine Engine (v3.0)")
    print("Launching on: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
