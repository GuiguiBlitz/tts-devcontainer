# Qwen3 TTS voice clone

- Code from https://huggingface.co/spaces/Qwen/Qwen3-TTS/blob/main/app.py
- Model from https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

## get the model

From the devcontainer run this to fetch the model
```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='./Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir_use_symlinks=False)"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='./Qwen3-TTS-12Hz-1.7B-Base', local_dir_use_symlinks=False)"
```

## Run the app
```bash
uv sync
uv run app.py
```