# coding=utf-8
# Qwen3-TTS Local Demo with Base and CustomVoice Models
import os
import gradio as gr
import numpy as np
import torch

# Local model paths
BASE_MODEL_PATH = "./Qwen3-TTS-12Hz-1.7B-Base"
CUSTOMVOICE_MODEL_PATH = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Global model holders
loaded_models = {}


def get_model(model_type):
    """Load a model by type (Base or CustomVoice)."""
    global loaded_models
    if model_type not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = BASE_MODEL_PATH if model_type == "Base" else CUSTOMVOICE_MODEL_PATH
        print(f"Loading {model_type} model from {model_path}...")
        loaded_models[model_type] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
        )
        print(f"{model_type} model loaded successfully!")
    return loaded_models[model_type]


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    
    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    
    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None
    
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    
    return None


# Speaker and language choices
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only):
    """Generate speech using Voice Clone with Base model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."
    
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."
    
    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."
    
    try:
        tts = get_model("Base")
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language if language != "Auto" else None,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."
    
    try:
        tts = get_model("CustomVoice")
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language if language != "Auto" else None,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    
    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """
    
    with gr.Blocks(title="Qwen3-TTS Local Demo") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Local Demo

Local Text-to-Speech demo with two models:
- **Voice Clone (Base)**: Clone any voice from a reference audio sample
- **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions

Models: `./Qwen3-TTS-12Hz-1.7B-Base` + `./Qwen3-TTS-12Hz-1.7B-CustomVoice`
"""
        )
        
        with gr.Tabs():
            # Tab 1: Voice Clone
            with gr.Tab("Voice Clone"):
                gr.Markdown("### Clone Voice from Reference Audio")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload a voice sample to clone)",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Text (Transcript of the reference audio)",
                            lines=2,
                            placeholder="Enter the exact text spoken in the reference audio...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (No reference text needed, but lower quality)",
                            value=False,
                        )
                    
                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                        )
                        clone_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                        clone_btn = gr.Button("Clone & Generate", variant="primary")
                
                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)
                
                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector],
                    outputs=[clone_audio_out, clone_status],
                )
            
            # Tab 2: TTS (CustomVoice)
            with gr.Tab("TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities."
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                        tts_instruct = gr.Textbox(
                            label="Style Instruction (Optional)",
                            lines=2,
                            placeholder="e.g., Speak in a cheerful and energetic tone",
                        )
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                    
                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)
                
                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct],
                    outputs=[tts_audio_out, tts_status],
                )
        
        gr.Markdown(
            """
---

**Using Local Models**: Base model for voice cloning, CustomVoice model for preset speakers. GPU acceleration recommended.
"""
        )
    
    return demo, theme, css


if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(share=False, theme=theme, css=css)