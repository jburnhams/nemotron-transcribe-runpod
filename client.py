"""Usage: python client.py audio.wav [--speakers]
Needs RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in the environment."""
import base64
import json
import os
import sys
import time
import urllib.request

key, ep = os.environ["RUNPOD_API_KEY"], os.environ["RUNPOD_ENDPOINT_ID"]
hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
base = f"https://api.runpod.ai/v2/{ep}"


def call(url, data=None):
    req = urllib.request.Request(url, data=data and json.dumps(data).encode(), headers=hdr)
    return json.load(urllib.request.urlopen(req))


audio = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
job = call(f"{base}/run", {"input": {"audio_base64": audio, "speakers": "--speakers" in sys.argv}})
while True:
    st = call(f"{base}/status/{job['id']}")
    if st["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        print(json.dumps(st.get("output", st), indent=2))
        break
    time.sleep(5)
