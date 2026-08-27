# Moving to Windows (RTX 3070 Ti) — Setup Procedure

This project currently runs on a Mac (CPU-only inference via `transformers`/`torch`, no CUDA).
Moving to a Windows desktop with an RTX 3070 Ti (8GB VRAM, Ampere) should make everything
faster and more reliable — no VRAM concerns for this 0.5B model. Follow these steps in order.

## 1. Install/update the NVIDIA driver

Install the latest Game Ready or Studio driver from NVIDIA for the RTX 3070 Ti. This alone
provides CUDA support — you do **not** need a separate "CUDA Toolkit" install; PyTorch and
onnxruntime ship their own CUDA runtime inside the pip wheel.

## 2. Install Python via Miniconda

Install Miniconda for Windows, then open an Anaconda Prompt and run:

```
conda create -n lovecraft python=3.10 -y
conda activate lovecraft
```

## 3. Install PyTorch with CUDA support (before anything else)

Do **not** just `pip install torch==2.13.0` — that may pull a CPU-only build. Go to
https://pytorch.org/get-started/locally/, select Windows / Pip / CUDA, and use the command
it gives you, e.g.:

```
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu124
```

(Use whatever CUDA tag the selector recommends for your currently installed driver — the
exact tag may differ from `cu124` by the time you do this.)

Verify it worked:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This should print `True` and `NVIDIA GeForce RTX 3070 Ti`.

## 4. Install FFmpeg (required by torchcodec)

`torchaudio.load()` routes through `torchcodec`, which needs FFmpeg's shared libraries —
and on Windows it specifically needs the **"full-shared"** build (not "essentials"), because
that's the one that ships the DLLs torchcodec dlopens at runtime.

Easiest path: via conda —

```
conda install -c conda-forge ffmpeg=7 -y
```

This installs it directly into the `lovecraft` env, which is where torchcodec looks for it
first (same approach used on the Mac). Alternative: download a "full-shared" build from
gyan.dev and add its `bin/` folder to your `PATH`.

## 5. Copy the project files over

From the Mac, copy these to the new machine (USB drive, network share, or `git`/cloud sync —
whatever's convenient):

- `main.py`
- `requirements.txt`

You do **not** need to copy the `CosyVoice/` folder — it has zero local modifications, so a
fresh clone on Windows is simpler (see step 6).

## 6. Clone CosyVoice

From inside the project folder on the Windows machine:

```
git clone https://github.com/FunAudioLLM/CosyVoice.git
```

No patching needed — it's used as-is.

## 7. Install Python dependencies — with one swap

Open `requirements.txt` and change:

```
onnxruntime==1.23.2
```

to:

```
onnxruntime-gpu==1.23.2
```

CosyVoice's frontend code auto-selects `CUDAExecutionProvider` whenever
`torch.cuda.is_available()` is `True` — with plain `onnxruntime` that provider doesn't exist,
and the speech tokenizer would silently fall back to CPU. `onnxruntime-gpu` needs CUDA/cuDNN
versions compatible with your driver; if installing or running it errors out, that's the next
thing to troubleshoot.

Then install everything:

```
pip install -r requirements.txt
```

If `pyworld` fails to build from source, you may need Microsoft C++ Build Tools installed
(Windows often lacks a C compiler by default) — install "Desktop development with C++" via
the Visual Studio Build Tools installer if that happens.

## 8. Get the model weights (9.1GB)

Either:

- **Copy directly** (faster if you have a fast transfer method): copy the whole
  `models/Fun-CosyVoice3-0.5B-2512/` folder from the Mac to the same relative path on Windows.
- **Re-download**:
  ```
  pip install "huggingface_hub[cli]"
  hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir models/Fun-CosyVoice3-0.5B-2512
  ```

Either way, the final path must be `<project_root>/models/Fun-CosyVoice3-0.5B-2512/`, matching
what `main.py` expects.

## 9. Run it

```
python main.py
```

It should launch noticeably faster than on the Mac, and `main.py` already auto-detects CUDA:

- `fp16=torch.cuda.is_available()` — automatically runs the LLM in fp16 for a speed boost
  on the GPU (this was a no-op on the Mac; it'll now be `True`).
- CosyVoice's own internal code (`model.py`, `frontend.py`) already auto-selects `cuda` over
  `cpu` when available — no changes needed there.

Open `http://127.0.0.1:7860`, upload/record a reference clip (auto-transcribed via Whisper),
enter target text, and generate.

## Known gotchas (things we hit on the Mac that may resurface differently on Windows)

- **`transformers` version matters a lot.** `requirements.txt` pins `transformers==4.51.3`
  deliberately — a newer major version (5.x) silently produces garbled/wrong-content audio
  with *no error or crash*, because it changes default model-loading/attention behavior. If
  you ever bump this package, and audio starts sounding like fluent gibberish in a random
  language, this pin is the first thing to check.
- **`<|endofprompt|>` marker.** Already handled in `main.py` — CosyVoice3 requires this
  literal marker in the prompt text or the LLM thread crashes. Not something you need to
  redo, just noting why that code is there.
- **torchcodec DLLs.** If you get `Could not load libtorchcodec` / `Library not loaded`
  errors, it's almost always the FFmpeg "full-shared" vs regular build issue from step 4.
- **CUDA/cuDNN version mismatches** between `torch`, `torchaudio`, and `onnxruntime-gpu` are
  the most likely source of new errors on this machine (unlike the Mac, which had no CUDA
  stack to get out of sync). If something fails to load a CUDA library, check that all three
  packages were built against compatible CUDA major versions.
