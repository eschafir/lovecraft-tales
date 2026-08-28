# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Gradio app (`main.py`) for zero-shot voice cloning using CosyVoice3. The user
uploads/records a reference audio clip, it's auto-transcribed with Whisper, and the target text
is spoken back in the cloned voice.

## Running

```
python main.py
```

Launches a Gradio UI at `http://127.0.0.1:7860`. There is no build step, lint config, or test
suite in this repo — it's one script.

## Architecture

- `main.py` is the entire application: it loads a Whisper model (`base`) for auto-transcribing
  the reference clip, loads `CosyVoice3` for inference, and wires both into a small Gradio
  `Blocks` UI (upload/record reference audio → transcribe → enter target text → generate).
- `CosyVoice/` (gitignored, not present until cloned) is a vendored, unmodified clone of
  https://github.com/FunAudioLLM/CosyVoice — `main.py` appends it and its
  `third_party/Matcha-TTS` subfolder to `sys.path` and imports `cosyvoice.cli.cosyvoice.CosyVoice3`
  directly from source rather than installing it as a package. There are zero local
  modifications to this folder; if it's missing, `git clone` it fresh into the project root.
- `models/Fun-CosyVoice3-0.5B-2512/` (gitignored) holds the model weights (~9.1GB), downloaded via
  `hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir models/Fun-CosyVoice3-0.5B-2512`.
  `main.py` expects this exact relative path.
- Device selection: `main.py` picks `mps` if available else `cpu` for the module-level `device`
  var, but the CosyVoice3 model itself and `fp16=` are driven by `torch.cuda.is_available()` —
  i.e. this codebase targets three environments (Mac/MPS, CPU, and CUDA on Windows) and the
  device logic is spread across a couple of independent checks rather than one unified device
  variable. Keep that in mind when changing device-selection code.
- CosyVoice3 requires the literal `<|endofprompt|>` marker somewhere in the prompt text passed
  to `inference_zero_shot`, or the LLM thread crashes; `main.py` injects a default one if the
  (possibly Whisper-transcribed) prompt text doesn't already contain it.
- `inference_zero_shot` returns a generator of streaming chunks; `main.py` concatenates all
  `tts_speech` tensors along dim=1 before returning audio to Gradio (no actual streaming to the
  UI currently).

## Dependency notes (see `requirements.txt` comments for the full list)

- `transformers` is pinned to `4.51.3` deliberately — 5.x silently produces garbled/wrong-content
  audio with no crash, because it changes default model-loading/attention behavior. If you ever
  need to bump this, verify actual audio output, not just that it imports.
- `torchcodec` (used by `torchaudio.load()`) requires FFmpeg's shared libraries at runtime. On
  Windows this specifically means the "full-shared" FFmpeg build, not "essentials".
- On a CUDA machine, `requirements.txt`'s `onnxruntime==1.23.2` needs to be swapped for
  `onnxruntime-gpu==1.23.2` — CosyVoice's frontend auto-selects `CUDAExecutionProvider` whenever
  `torch.cuda.is_available()` is `True`, and plain `onnxruntime` doesn't have that provider (silent
  CPU fallback for the speech tokenizer, not a crash).
- `setuptools<81` is pinned because `pyworld` imports `pkg_resources` at runtime, which
  `setuptools>=81` removed.

## Platform migration

`windows-procedure.md` documents the full Mac → Windows (CUDA) migration steps this project has
gone through, including gotchas already hit once (transformers version, torchcodec DLLs,
CUDA/cuDNN mismatches). Check it before re-deriving setup steps for a new machine or debugging
environment-specific failures.
