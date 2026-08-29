#!/usr/bin/env python3
"""
Convert H. P. Lovecraft tales into audiobooks using CosyVoice3 zero-shot voice cloning.

Features:
- Clones voice from reference audio (e.g. Vincent Price Voice.mp3).
- Uses Whisper for one-time reference transcription (cached to .txt).
- Smart TTS text chunking along natural sentence and paragraph boundaries (0 overlap).
- Combines title & author into a smooth introductory narration.
- Inserts natural pauses between sentences (350ms), paragraphs (800ms), and chapters (1500ms).
- Checkpointed / resumable generation (saves intermediate chunk WAVs in results/<tale>/).
- Stitches chunks into a complete master audio file (results/<tale>.wav).
- Supports single tale, batch processing, and progress bars.
"""

import argparse
import glob
import os
import re
import sys
import time
import numpy as np
import soundfile as sf
import torch

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

try:
    from cosyvoice.cli.cosyvoice import CosyVoice3
except ImportError as e:
    print(f"Error importing CosyVoice3: {e}", file=sys.stderr)
    print("Please make sure you are in the 'lovecraft' conda environment.", file=sys.stderr)
    sys.exit(1)

import whisper
from tqdm import tqdm


def clean_markdown_for_speech(md_content: str) -> list[dict]:
    """
    Clean markdown formatting and split into natural spoken chunks with pause durations.
    Returns a list of dicts: [{"text": "...", "pause_ms": 300}, ...]
    """
    lines = md_content.splitlines()
    title = ""
    author = ""
    story_lines = []

    # Extract title and author from top of markdown if present
    for line in lines:
        s = line.strip()
        if not title and s.startswith("# "):
            title = s.lstrip("# ").strip()
            continue
        if title and not author and ("By H. P. Lovecraft" in s or "By " in s):
            author = re.sub(r"[*_]", "", s).strip()
            continue
        if s == "---":
            continue
        story_lines.append(line)

    cleaned_lines = []
    for line in story_lines:
        stripped = line.strip()
        if not stripped or stripped == "---":
            cleaned_lines.append("")
            continue

        # Convert section/chapter headers
        if stripped.startswith("#"):
            header_text = re.sub(r"^#+\s*", "", stripped)
            m_roman = re.match(r"^([IVXLCDM]+)\.\s*(.*)", header_text, re.IGNORECASE)
            if m_roman:
                roman_num, rest = m_roman.groups()
                line_clean = f"Chapter {roman_num}. {rest}".strip()
            else:
                line_clean = header_text
            
            if not line_clean.endswith((".", "!", "?", ":")):
                line_clean += "."
            cleaned_lines.append(f"__CHAPTER_PAUSE__{line_clean}")
            continue

        # Strip bold/italic/quote markdown markers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"^>\s*", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        cleaned_lines.append(text)

    # Group into paragraphs
    raw_full = "\n".join(cleaned_lines)
    paragraphs = [p.strip() for p in raw_full.split("\n\n") if p.strip()]

    chunks = []
    max_chunk_chars = 280  # Ideal length for CosyVoice flow model

    # Add introduction chunk if title and author found
    if title:
        intro = f"{title}, {author}." if author else f"{title}."
        chunks.append({"text": intro, "pause_ms": 1200})

    for p in paragraphs:
        is_chapter = False
        if p.startswith("__CHAPTER_PAUSE__"):
            is_chapter = True
            p = p.replace("__CHAPTER_PAUSE__", "")

        # Split paragraph into sentences using punctuation boundaries
        sentences = re.split(r"(?<=[.!?])\s+", p)
        current_chunk = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            if len(current_chunk) + len(s) + 1 <= max_chunk_chars:
                current_chunk = (current_chunk + " " + s).strip()
            else:
                if current_chunk:
                    chunks.append({"text": current_chunk, "pause_ms": 350})
                # If a single sentence exceeds max_chunk_chars, split on commas/semicolons
                if len(s) > max_chunk_chars:
                    sub_parts = re.split(r"(?<=[,;—–])\s+", s)
                    sub_acc = ""
                    for sub in sub_parts:
                        if len(sub_acc) + len(sub) + 1 <= max_chunk_chars:
                            sub_acc = (sub_acc + " " + sub).strip()
                        else:
                            if sub_acc:
                                chunks.append({"text": sub_acc, "pause_ms": 300})
                            sub_acc = sub
                    if sub_acc:
                        current_chunk = sub_acc
                else:
                    current_chunk = s

        if current_chunk:
            pause = 1500 if is_chapter else 800
            chunks.append({"text": current_chunk, "pause_ms": pause})

    return chunks


def get_or_create_transcript(voice_audio_path: str, transcript_override: str = None) -> str:
    """
    Get prompt audio transcript. Uses cached .txt if available, otherwise transcribes with Whisper.
    """
    if transcript_override and os.path.exists(transcript_override):
        with open(transcript_override, "r", encoding="utf-8") as f:
            return f.read().strip()

    base_name = os.path.splitext(voice_audio_path)[0]
    cached_txt = f"{base_name}_transcript.txt"

    if os.path.exists(cached_txt):
        with open(cached_txt, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
            if transcript:
                print(f"Loaded cached voice transcript from: {cached_txt}")
                return transcript

    print(f"Transcribing reference audio '{voice_audio_path}' with Whisper...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(voice_audio_path)
    transcript = result["text"].strip()

    with open(cached_txt, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Saved reference transcript to: {cached_txt}")
    print(f"Transcript: \"{transcript}\"")
    return transcript


def synthesize_tale(
    tale_path: str,
    cosyvoice: CosyVoice3,
    prompt_speech: str,
    prompt_text: str,
    output_dir: str = "results",
    skip_existing: bool = True,
    keep_chunks: bool = True,
) -> str:
    """
    Synthesizes a single tale Markdown file into a master WAV audio file.
    Returns path to the generated master audio file.
    """
    tale_name = os.path.splitext(os.path.basename(tale_path))[0]
    tale_dir = os.path.join(output_dir, tale_name)
    os.makedirs(tale_dir, exist_ok=True)
    master_file = os.path.join(tale_dir, f"{tale_name}.wav")

    if skip_existing and os.path.exists(master_file):
        print(f"\n[SKIP] Master file already exists: {master_file}")
        return master_file

    tale_chunk_dir = tale_dir

    with open(tale_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    chunks = clean_markdown_for_speech(md_content)
    if not chunks:
        print(f"[WARN] No readable chunks found in {tale_path}")
        return ""

    sample_rate = cosyvoice.sample_rate
    audio_segments = []

    # Ensure <|endofprompt|> tag is present
    if "<|endofprompt|>" not in prompt_text:
        formatted_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
    else:
        formatted_prompt = prompt_text

    print(f"\nNarrating '{tale_name}' ({len(chunks)} chunks)...")
    progress_bar = tqdm(enumerate(chunks), total=len(chunks), desc=tale_name, unit="chunk")

    for idx, chunk_info in progress_bar:
        chunk_text = chunk_info["text"]
        pause_ms = chunk_info["pause_ms"]
        chunk_file = os.path.join(tale_chunk_dir, f"part_{idx:04d}.wav")

        # Resumability check
        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1000:
            audio_data, _ = sf.read(chunk_file, dtype="float32")
        else:
            try:
                output = cosyvoice.inference_zero_shot(chunk_text, formatted_prompt, prompt_speech)
                audio_tensors = [c["tts_speech"] for c in output]
                if not audio_tensors:
                    print(f"\n[WARN] No audio returned for chunk {idx}: {chunk_text}")
                    continue
                audio_tensor = torch.cat(audio_tensors, dim=1).squeeze().cpu().numpy()
                sf.write(chunk_file, audio_tensor, sample_rate)
                audio_data = audio_tensor
            except Exception as e:
                print(f"\n[ERROR] Failed on chunk {idx} ('{chunk_text[:50]}...'): {e}", file=sys.stderr)
                continue

        audio_segments.append(audio_data)

        # Append silence for natural pause
        if pause_ms > 0:
            pause_samples = int(sample_rate * (pause_ms / 1000.0))
            silence = np.zeros(pause_samples, dtype=np.float32)
            audio_segments.append(silence)

    if not audio_segments:
        print(f"[ERROR] No audio generated for {tale_name}", file=sys.stderr)
        return ""

    # Stitch all chunks into the master audiobook
    master_audio = np.concatenate(audio_segments)
    sf.write(master_file, master_audio, sample_rate)
    
    total_seconds = len(master_audio) / sample_rate
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    print(f"Finished '{tale_name}'! Duration: {minutes}m {seconds}s -> {master_file}")

    if not keep_chunks:
        import shutil
        shutil.rmtree(tale_chunk_dir, ignore_errors=True)

    return master_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lovecraft tales into audiobooks using CosyVoice3 zero-shot voice cloning."
    )
    parser.add_argument(
        "--tale",
        type=str,
        default=None,
        help="Path or name of a specific tale to generate (e.g. 'dagon' or 'tales/dagon.md').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate audio for all tales found in tales/ directory.",
    )
    parser.add_argument(
        "--tales-dir",
        type=str,
        default="tales",
        help="Directory containing tale Markdown files (default: 'tales').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("Results", "Audio"),
        help="Directory to save generated audiobooks (default: 'Results/Audio').",
    )
    parser.add_argument(
        "--voice-sample",
        type=str,
        default="Vincent Price Voice.mp3",
        help="Reference audio sample for voice cloning (default: 'Vincent Price Voice.mp3').",
    )
    parser.add_argument(
        "--voice-transcript",
        type=str,
        default=None,
        help="Optional text transcript of the voice sample. If omitted, Whisper is used.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "models", "Fun-CosyVoice3-0.5B-2512"),
        help="Path to CosyVoice3 model weights directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tales to process in batch mode (for testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate master audio even if it already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tales and exit.",
    )

    args = parser.parse_args()

    # List tales mode
    all_tale_files = sorted(glob.glob(os.path.join(args.tales_dir, "*.md")))
    if args.list:
        print(f"\nAvailable tales in '{args.tales_dir}' ({len(all_tale_files)} found):")
        for f in all_tale_files:
            name = os.path.splitext(os.path.basename(f))[0]
            print(f"  - {name}")
        return

    # Determine files to process
    if args.tale:
        target = args.tale
        if not target.endswith(".md"):
            target = os.path.join(args.tales_dir, f"{target}.md")
        if not os.path.exists(target):
            print(f"[ERROR] Tale file not found: {target}", file=sys.stderr)
            sys.exit(1)
        files_to_process = [target]
    elif args.all:
        files_to_process = all_tale_files
        if not files_to_process:
            print(f"[ERROR] No .md files found in '{args.tales_dir}'", file=sys.stderr)
            sys.exit(1)
    else:
        print("Please specify either --tale <name> or --all to process tales.")
        print("Run 'python generate_audio.py --help' or 'python generate_audio.py --list' for details.")
        return

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    # Verify voice sample
    voice_path = os.path.abspath(args.voice_sample)
    if not os.path.exists(voice_path):
        print(f"[ERROR] Reference voice file not found: {voice_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Get or create reference transcript
    prompt_text = get_or_create_transcript(voice_path, args.voice_transcript)

    # 2. Load CosyVoice3 model
    print(f"\nLoading CosyVoice3 model from: {args.model_dir}...")
    use_cuda = torch.cuda.is_available()
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if use_cuda else 'CPU'}")
    
    cosyvoice = CosyVoice3(args.model_dir, fp16=use_cuda)
    print("CosyVoice3 loaded successfully!\n")

    print(f"Processing {len(files_to_process)} tale(s)...")
    start_time = time.time()

    for idx, tale_file in enumerate(files_to_process, 1):
        print(f"\n[{idx}/{len(files_to_process)}] Processing: {os.path.basename(tale_file)}")
        synthesize_tale(
            tale_path=tale_file,
            cosyvoice=cosyvoice,
            prompt_speech=voice_path,
            prompt_text=prompt_text,
            output_dir=args.output_dir,
            skip_existing=not args.force,
        )

    elapsed = time.time() - start_time
    print(f"\nAll tasks complete in {elapsed/60:.1f} minutes! Results saved to '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
