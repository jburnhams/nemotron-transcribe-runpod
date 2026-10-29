"""RunPod serverless handler: starts a local vLLM server for Nemotron 3.5 Transcribe
and forwards transcription requests to it."""
import base64
import json
import os
import subprocess
import time
import urllib.request

import runpod

MODEL = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_S = int(os.environ.get("STARTUP_TIMEOUT_S", "900"))
DEFAULT_PROMPT = "Transcribe the following audio."
SPEAKER_PROMPT = (
    "Transcribe the following audio with punctuation, capitalization and speaker tags. "
    "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)

_server = None


def _start_server():
    global _server
    if _server is not None:
        return
    env = dict(
        os.environ,
        VLLM_PLUGINS="nemo_speechlm",
        VLLM_MAX_AUDIO_DECODE_DURATION_S="3600",
        VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="180",
        VLLM_USE_V2_MODEL_RUNNER="0",
    )
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "65536"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEM_UTIL", "0.90"),
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":8}',
        "--trust-remote-code",
    ]
    if "FP8" not in MODEL and "NVFP4" not in MODEL:
        cmd += ["--dtype", "bfloat16"]
    _server = subprocess.Popen(cmd, env=env)
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(f"vllm exited with code {_server.returncode}")
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("vllm did not become healthy in time")


def _audio_data_url(inp):
    if inp.get("audio_base64"):
        return f"data:audio/wav;base64,{inp['audio_base64']}"
    if inp.get("audio_url"):
        with urllib.request.urlopen(inp["audio_url"], timeout=120) as r:
            return "data:audio/wav;base64," + base64.b64encode(r.read()).decode()
    raise ValueError("provide audio_base64 or audio_url")


def handler(job):
    inp = job["input"]
    try:
        _start_server()
        prompt = inp.get("prompt") or (SPEAKER_PROMPT if inp.get("speakers") else DEFAULT_PROMPT)
        body = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": int(inp.get("max_tokens", 16384)),
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": _audio_data_url(inp)}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        req = urllib.request.Request(
            f"{BASE}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.load(r)
        return {"text": out["choices"][0]["message"]["content"], "model": MODEL}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
