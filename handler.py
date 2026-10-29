"""RunPod serverless handler: starts a local vLLM server for Nemotron 3.5 Transcribe
and forwards transcription requests to it."""
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import runpod

MODEL = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_S = int(os.environ.get("STARTUP_TIMEOUT_S", "900"))
DEFAULT_PROMPT = "Transcribe the following audio."
SPEAKER_PROMPT = (
    "Transcribe the following audio with punctuation, capitalization and speaker tags. "
    "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)

_server = None
LOG_PATH = "/tmp/vllm.log"


def _model_path():
    """Use RunPod's pre-cached copy of the model if the endpoint's Model field is set."""
    folder = os.path.join(CACHE_ROOT, "models--" + MODEL.replace("/", "--"))
    try:
        with open(os.path.join(folder, "refs", "main")) as f:
            snap = os.path.join(folder, "snapshots", f.read().strip())
        if os.path.isdir(snap):
            return snap
    except OSError:
        pass
    return MODEL


def _log_tail(n=4000):
    try:
        with open(LOG_PATH) as f:
            return f.read()[-n:]
    except OSError:
        return ""


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
        "vllm", "serve", _model_path(), "--served-model-name", MODEL,
        "--port", str(PORT),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "65536"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEM_UTIL", "0.90"),
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":8}',
        "--trust-remote-code",
    ]
    if "FP8" not in MODEL and "NVFP4" not in MODEL:
        cmd += ["--dtype", "bfloat16"]
    _log = open(LOG_PATH, "w")
    _server = subprocess.Popen(cmd, env=env, stdout=_log, stderr=subprocess.STDOUT)
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if _server.poll() is not None:
            code, _server = _server.returncode, None
            raise RuntimeError(f"vllm exited with code {code}:\n{_log_tail()}")
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("vllm did not become healthy in time")


TOKENS_PER_SECOND = float(os.environ.get("TOKENS_PER_SECOND", "10"))  # ~4 words/s plus headroom
MAX_REPEAT_NGRAM = 6
MIN_REPEATS = 4


def _audio_bytes(inp):
    if inp.get("audio_base64"):
        return base64.b64decode(inp["audio_base64"])
    if inp.get("audio_url"):
        with urllib.request.urlopen(inp["audio_url"], timeout=120) as r:
            return r.read()
    raise ValueError("provide audio_base64 or audio_url")


def _duration_s(data):
    """Audio length via ffprobe on a temp file, or None if it cannot be determined."""
    try:
        with tempfile.NamedTemporaryFile() as f:
            f.write(data)
            f.flush()
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", f.name],
                capture_output=True, timeout=60,
            )
        return float(p.stdout.decode().strip())
    except Exception:
        return None


def _collapse_repeats(text):
    """Cut runs where the same 1..6 word phrase repeats MIN_REPEATS or more times in a row,
    keeping two copies. Returns (text, trimmed)."""
    words = text.split()
    out, i, trimmed = [], 0, False
    while i < len(words):
        for n in range(1, MAX_REPEAT_NGRAM + 1):
            gram = [w.lower().strip(".,!?") for w in words[i:i + n]]
            if len(gram) < n:
                continue
            reps = 1
            while [w.lower().strip(".,!?") for w in words[i + reps * n:i + (reps + 1) * n]] == gram:
                reps += 1
            if reps >= MIN_REPEATS:
                out.extend(words[i:i + 2 * n])
                i += reps * n
                trimmed = True
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out), trimmed


def handler(job):
    inp = job["input"]
    try:
        _start_server()
        data = _audio_bytes(inp)
        prompt = inp.get("prompt") or (SPEAKER_PROMPT if inp.get("speakers") else DEFAULT_PROMPT)
        max_tokens = int(inp.get("max_tokens", 16384))
        dur = _duration_s(data)
        if dur:
            max_tokens = min(max_tokens, int(dur * TOKENS_PER_SECOND) + 64)
        body = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url",
                     "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(data).decode()}},
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
        choice = out["choices"][0]
        text, trimmed = _collapse_repeats(choice["message"]["content"])
        return {
            "text": text,
            "model": MODEL,
            "duration_s": dur,
            "hit_token_cap": choice.get("finish_reason") == "length",
            "loop_trimmed": trimmed,
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
