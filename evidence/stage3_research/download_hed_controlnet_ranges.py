"""Resume the official TheDenk hed ControlNet safetensors in verified ranges."""
import concurrent.futures as futures
import hashlib
import math
import os
import time
from pathlib import Path

import requests

URL = "https://huggingface.co/TheDenk/wan2.2-ti2v-5b-controlnet-hed-v1/resolve/main/diffusion_pytorch_model.safetensors"
SIZE = 691_979_056
SHA256 = "034b869a5dd60db18170d33008e50b8d795d55b978e3ded85a8cb93f448293b0"
CHUNK = 4 << 20
WORKERS = 16
DEST = Path(__file__).resolve().parents[1] / "paper_sources/wan2.2-controlnet-weights/hed"
DEST.mkdir(parents=True, exist_ok=True)
PARTS = DEST / "ranges"
PARTS.mkdir(exist_ok=True)
TARGET = DEST / "diffusion_pytorch_model.safetensors"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(8 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def download_one(index):
    begin = index * CHUNK
    end = min(SIZE, begin + CHUNK) - 1
    part = PARTS / f"{index:04d}.part"
    if part.exists() and part.stat().st_size == end - begin + 1:
        return index
    for attempt in range(12):
        try:
            with requests.get(URL, headers={"Range": f"bytes={begin}-{end}"}, stream=True, timeout=(20, 150)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {begin}-{end}":
                    raise RuntimeError((response.status_code, response.headers.get("Content-Range")))
                temp = part.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    for data in response.iter_content(1 << 20):
                        stream.write(data)
                if temp.stat().st_size != end - begin + 1:
                    raise RuntimeError((index, temp.stat().st_size))
                os.replace(temp, part)
                return index
        except (requests.RequestException, RuntimeError) as exc:
            print("retry", index, attempt + 1, str(exc)[:120], flush=True)
            time.sleep(min(30, attempt * 2 + 1))
    raise RuntimeError(f"failed range {index}")


if __name__ == "__main__":
    n = math.ceil(SIZE / CHUNK)
    print("ranges", n, "workers", WORKERS, flush=True)
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, _ in enumerate(pool.map(download_one, range(n)), 1):
            if done % 8 == 0 or done == n:
                print("completed", done, "/", n, flush=True)
    with TARGET.open("wb") as out:
        for i in range(n):
            with (PARTS / f"{i:04d}.part").open("rb") as part:
                for data in iter(lambda: part.read(8 << 20), b""):
                    out.write(data)
    assert TARGET.stat().st_size == SIZE
    assert sha256(TARGET) == SHA256
    print("SHA256 verified", TARGET, SHA256, flush=True)
