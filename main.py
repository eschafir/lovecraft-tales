import gradio as gr
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import torch
import torchaudio
import whisper

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_DIR = os.path.join(ROOT_DIR, "CosyVoice")
sys.path.append(COSYVOICE_DIR)
sys.path.append(os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS"))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from cosyvoice.cli.cosyvoice import CosyVoice3

# Local path (downloaded via `hf download`) to avoid ModelScope's CDN
MODEL_DIR = os.path.join(ROOT_DIR, "models", "Fun-CosyVoice3-0.5B-2512")

# Initialize model
cosyvoice = CosyVoice3(MODEL_DIR)

# Used to auto-transcribe the reference audio sample
whisper_model = whisper.load_model("base")

def transcribe_reference(prompt_speech):
    if not prompt_speech:
        return ""
    result = whisper_model.transcribe(prompt_speech)
    return result["text"].strip()

def generate_voice(prompt_speech, prompt_text, target_text):
    if not prompt_speech or not target_text:
        return None

    # CosyVoice3 requires an <|endofprompt|> marker in the prompt text
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text

    # Run zero-shot voice cloning inference
    output = cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_speech)
    
    # Concatenate streaming chunks
    audio_tensors = [chunk['tts_speech'] for chunk in output]
    final_audio = torch.cat(audio_tensors, dim=1)
    
    return (cosyvoice.sample_rate, final_audio.squeeze().cpu().numpy())

# Build UI
with gr.Blocks(title="CosyVoice Zero-Shot TTS") as demo:
    gr.Markdown("# 🎙️ CosyVoice Voice Cloning Studio")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio Sample")
            prompt_text = gr.Textbox(label="Reference Audio Transcript", placeholder="Auto-transcribed from the reference audio, or type your own...")
            target_text = gr.Textbox(label="Text to Speak", lines=4, placeholder="Enter what you want the cloned voice to say...")
            submit_btn = gr.Button("Clone & Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio", autoplay=True)
            
    audio_input.change(
        fn=transcribe_reference,
        inputs=audio_input,
        outputs=prompt_text
    )

    submit_btn.click(
        fn=generate_voice,
        inputs=[audio_input, prompt_text, target_text],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch()