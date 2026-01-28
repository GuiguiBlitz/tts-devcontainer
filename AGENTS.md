# Project Context: Qwen3-TTS Voice Cloning

## 1. Mission
This project implements a local Voice Cloning interface using the **Qwen3-TTS-12Hz-1.7B-CustomVoice** model. It uses a strict `devcontainer` environment with `uv` for package management and `Gradio` for the frontend.

## 2. Environment & Constraints
* **Package Manager**: `uv` (Strictly used for all dependency management).
    * Add packages: `uv add <package>`
    * Run scripts: `uv run python <script.py>`
* **Python Version**: 3.10
* **Hardware**: NVIDIA GPU (CUDA) required.
* **Model Storage**:
    * Models are stored strictly in the local directory (e.g., `./Qwen3-TTS-12Hz-1.7B-CustomVoice`).
    * Git must **IGNORE** these model folders (`.gitignore` enforcement).
    * Symlinks are disabled (`local_dir_use_symlinks=False`).

## 3. Logical Agents (System Components)

### 🤖 The Model Manager (Inference Engine)
* **Responsibility**: Loading the heavy 1.7B parameter model onto the GPU safely.
* **Critical Instruction**: 
    * Must verify `model.safetensors` exists in the local path before loading.
    * Uses `bfloat16` and `flash_attention_2` for optimization.
* **Source**: `qwen_tts` library.

### 🗣️ The Synthesizer
* **Responsibility**: converting text + prompt + speaker ID into raw audio waveforms.
* **Inputs**:
    * `text` (The content).
    * `speaker` (The voice profile).
    * `instruct` (Style/Emotion).
* **Output**: `.wav` file path or numpy array.

### 🖥️ The Interface Agent (Gradio UI)
* **Responsibility**: User interaction.
* **Structure**:
    * **Input Column**: Text box, Dropdowns (Speaker, Language), Instruction box.
    * **Output Column**: Audio player.
* **File**: `app.py`

## 4. Development Rules
1.  **Dependency Changes**: Never use `pip install` directly. Always use `uv add` or modify `pyproject.toml` and run `uv sync`.
2.  **Pathing**: Always use relative paths for the model (`./Qwen...`).
3.  **Hugging Face Interactions**: Use the Python API (`huggingface_hub`) for downloads to ensure stability over CLI tools.

## 5. Current Directory Structure
```text
/workspaces/voice-clone/
├── .devcontainer/       # Docker config
├── .venv/               # Managed by uv
├── Qwen3-TTS.../        # Model Artifacts (Ignored by Git)
├── app.py               # Main Application
├── pyproject.toml       # Dependencies
├── uv.lock              # Lockfile
└── agents.md            # This file