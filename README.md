# nemotron-transcribe-runpod

RunPod serverless endpoint that transcribes audio with NVIDIA Nemotron 3.5 Transcribe
(vLLM + NeMo Speech). **Untested**: written without GPU access, expect a first-deploy fix or two.

## Deploy
1. Push to `main`; GitHub Actions builds `ghcr.io/jburnhams/nemotron-transcribe-runpod:latest`.
   Make the package public (or add registry credentials in RunPod).
2. RunPod > Serverless > New Endpoint > custom image above.
   - GPU: 80 GB (A100/H100) for BF16. FP8/NVFP4 repos were not public yet at release (set `MODEL_ID` once they are, then 48 GB works).
   - Set the endpoint's **Model** field to `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` so RunPod pre-caches the weights on the host (no volume, no storage cost). The handler finds them under `/runpod-volume/huggingface-cache/hub`, and falls back to downloading from Hugging Face.
   - Container disk 30 GB+, execution timeout 3600 s, idle timeout ~5 s, max workers 1 to start.
   - Optional env: `MODEL_ID`, `GPU_MEM_UTIL`.
3. Cold start downloads and loads the model (several minutes the first time).

## Use
```
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
python client.py clip.wav            # plain transcript
python client.py clip.wav --speakers # speaker tags
```
Request input: `audio_base64` or `audio_url`, optional `prompt`, `speakers`, `max_tokens`.
Response: `{"text": "...", "model": "..."}`. RunPod `/run` bodies are capped at ~10 MB, so for
long audio pass `audio_url` (e.g. a presigned link) instead.

## Loop prevention
Output length is capped from the audio duration (`TOKENS_PER_SECOND`, default 10), runs of a phrase repeated 4+ times are collapsed to two copies, and the response includes `hit_token_cap` and `loop_trimmed` flags so you can spot suspect transcripts.
