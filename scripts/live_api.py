"""M2c S2: the API the live-boxes app talks to (docs/m2c-live-boxes.md, "Serving it to the phone").

    python scripts/live_api.py [--with-store] [--port 8765]

Python's standard library on 127.0.0.1 only; Vite (web/) proxies /api to
it, so the phone reaches it only through the HTTPS page. It answers only
requests carrying the key it prints at start (header X-Live-Key). The page
gets the key from the URL's fragment (https://<laptop>:5180/#k=<key>), which
is never sent in a request line, so it isn't in any log.

    GET  /api/config                    threshold, classes, the sets served, the model files
    GET  /api/models/<variant>/<sha256> a model file, cached by its hash
    GET  /api/bench/<set>               the set's photos, by index and size only
    GET  /api/bench/<set>/<i>           photo i as stored (JPEG); X-Decoded-Sha256 is the
                                        SHA-256 of its pixels as PIL decodes them (RGB)
    GET  /api/bench/<set>/<i>/input     Ultralytics' LetterBox of photo i, 1,280 square: raw
                                        RGB bytes, for the bench's pixel check
    POST /api/records/<kind>            a run record (bench-parity, bench-speed, live, motion, or
                                        error: what failed, since nothing else of the phone's is seen),
                                        written with the API's git commit, the models' hashes and
                                        the time into this start's run folder

Store photos never reach the phone: v1-val (the store's validation photos)
is served only with --with-store, and then only to the laptop's own browser:
Vite's proxy adds the browser's address (X-Forwarded-For, its `xfwd`), and
a store set goes only to a request whose last hop is this laptop. Without
--with-store the API refuses it to everyone. real-val is ExpDate's (public). Photos are named by set
and index only. Records go to results/runs/m2c/<time>-m2c-app/.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import socket
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HOST = "127.0.0.1"
PORT = 8765
WEB_PORT = 5180
KEY_HEADER = "X-Live-Key"
PUBLIC_SETS = ("real-val",)          # ExpDate's: may go to the phone
STORE_SETS = ("v1-val",)             # the retailer's: the laptop's browser only, with --with-store
RECORD_KINDS = ("bench-parity", "bench-speed", "live", "motion", "error")   # error: what failed on the phone
MAX_BODY = 64 * 1024 * 1024          # a parity record of 165 photos is a few MB
IMGSZ = 1280

_MODEL = re.compile(r"/api/models/([a-z0-9-]{1,32})/([0-9a-f]{64})")
_SET = re.compile(r"/api/bench/([a-z0-9-]{1,32})")
_PHOTO = re.compile(r"/api/bench/([a-z0-9-]{1,32})/(\d{1,5})")
_INPUT = re.compile(r"/api/bench/([a-z0-9-]{1,32})/(\d{1,5})/input")
_RECORD = re.compile(r"/api/records/([a-z-]{1,32})")
_LABEL = re.compile(r"[a-z0-9-]{1,32}")


@dataclass(frozen=True)
class Photo:
    width: int
    height: int
    path: Path


@dataclass
class Api:
    """What the handler serves. Built once at start (make_api); every name a
    request can reach is a key of `models` or `sets`, never a path."""
    key: str
    models: dict[str, dict]              # variant: {"path", "sha256", "bytes"}
    sets: dict[str, list[Photo]]         # served sets only
    refused: tuple[str, ...]             # sets known but not served (store, without --with-store)
    run_dir: Path
    config: dict                         # threshold, classes, imgsz: sent to the app
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _decoded: dict = field(default_factory=dict)

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq


def decoded_sha256(path: Path) -> str:
    """SHA-256 of a photo's pixels as the scoring decodes them (load_upright:
    PIL, EXIF turn applied, RGB), row by row."""
    import numpy as np

    from bestbefore.images import load_upright

    return hashlib.sha256(np.ascontiguousarray(np.asarray(load_upright(path))).tobytes()).hexdigest()


def letterboxed_rgb(path: Path) -> bytes:
    """Ultralytics' LetterBox of the photo, as its predictor makes the model's
    input at 1,280 square (grey 114, INTER_LINEAR), as raw RGB bytes."""
    import numpy as np
    from ultralytics.data.augment import LetterBox

    from bestbefore.images import load_upright

    out = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)(image=np.asarray(load_upright(path)))
    return np.ascontiguousarray(out).tobytes()


class Handler(BaseHTTPRequestHandler):
    api: Api                             # set on the subclass make_server builds
    server_version = "live-api"
    sys_version = ""

    # --- plumbing ---------------------------------------------------------------------

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/json",
              headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        headers = headers or {}
        headers.setdefault("Cache-Control", "no-store")
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode())

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _authorised(self) -> bool:
        given = self.headers.get(KEY_HEADER, "")
        if hmac.compare_digest(given.encode(), self.api.key.encode()):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, f"no valid {KEY_HEADER}")
        return False

    def _from_laptop(self) -> bool:
        """Whether the request came from this laptop: the last address in
        X-Forwarded-For (the one Vite's proxy added) is a loopback one, or,
        with none, the connection's own (always loopback: the API listens on
        127.0.0.1 only)."""
        hops = [h.strip() for h in self.headers.get("X-Forwarded-For", "").split(",") if h.strip()]
        last = hops[-1] if hops else self.client_address[0]
        try:
            return ipaddress.ip_address(last.removeprefix("::ffff:")).is_loopback
        except ValueError:
            return False

    def _served(self) -> list[str]:
        """The sets this request may have: store sets to the laptop only."""
        return [s for s in self.api.sets if s not in STORE_SETS or self._from_laptop()]

    def _photos(self, name: str) -> list[Photo] | None:
        """The set's photos, or None after answering 403 (a store set not
        served, or asked for by another device) or 404 (no such set)."""
        if name in self._served():
            return self.api.sets[name]
        if name in self.api.refused:
            self._error(HTTPStatus.FORBIDDEN, f"{name} is served only with --with-store, to the laptop's browser")
        elif name in self.api.sets:
            self._error(HTTPStatus.FORBIDDEN, f"{name} is served to the laptop's own browser only")
        else:
            self._error(HTTPStatus.NOT_FOUND, "no such set")
        return None

    def _photo(self, name: str, index: str) -> Photo | None:
        photos = self._photos(name)
        if photos is None:
            return None
        if int(index) >= len(photos):
            self._error(HTTPStatus.NOT_FOUND, "no such photo")
            return None
        return photos[int(index)]

    # --- requests ---------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server's name)
        if not self._authorised():
            return
        path = urlsplit(self.path).path
        if path == "/api/config":
            return self._json({**self.api.config, "sets": sorted(self._served()),
                               "models": {v: {"sha256": m["sha256"], "bytes": m["bytes"],
                                              "url": f"/api/models/{v}/{m['sha256']}"}
                                          for v, m in self.api.models.items()},
                               "run": self.api.run_dir.name})
        if m := _MODEL.fullmatch(path):
            model = self.api.models.get(m.group(1))
            if model is None or model["sha256"] != m.group(2):
                return self._error(HTTPStatus.NOT_FOUND, "no such model file")
            return self._send(HTTPStatus.OK, model["path"].read_bytes(), "application/octet-stream",
                              {"Cache-Control": "private, max-age=31536000, immutable",
                               "ETag": f'"{model["sha256"]}"'})
        if m := _SET.fullmatch(path):
            photos = self._photos(m.group(1))
            if photos is not None:
                self._json({"set": m.group(1),
                            "photos": [{"index": i, "width": p.width, "height": p.height}
                                       for i, p in enumerate(photos)]})
            return None
        if m := _PHOTO.fullmatch(path):
            photo = self._photo(*m.groups())
            if photo is not None:
                key = (m.group(1), int(m.group(2)))
                if key not in self.api._decoded:
                    self.api._decoded[key] = decoded_sha256(photo.path)
                self._send(HTTPStatus.OK, photo.path.read_bytes(), "image/jpeg",
                           {"Cache-Control": "private, max-age=86400", "X-Decoded-Sha256": self.api._decoded[key]})
            return None
        if m := _INPUT.fullmatch(path):
            photo = self._photo(*m.groups())
            if photo is not None:
                self._send(HTTPStatus.OK, letterboxed_rgb(photo.path), "application/octet-stream",
                           {"Cache-Control": "private, max-age=86400",
                            "X-Input-Width": str(IMGSZ), "X-Input-Height": str(IMGSZ)})
            return None
        return self._error(HTTPStatus.NOT_FOUND, "no such route")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorised():
            return
        m = _RECORD.fullmatch(urlsplit(self.path).path)
        if not m or m.group(1) not in RECORD_KINDS:
            return self._error(HTTPStatus.NOT_FOUND, "no such record kind")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length else HTTPStatus.LENGTH_REQUIRED,
                               f"a record is 1 byte to {MAX_BODY} bytes")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "not JSON")
        if not isinstance(body, dict):
            return self._error(HTTPStatus.BAD_REQUEST, "a record is a JSON object")
        name = write_record(self.api, m.group(1), body)
        return self._json({"file": name, "run": self.api.run_dir.name}, HTTPStatus.CREATED)

    def do_PUT(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "GET or POST")

    do_DELETE = do_PATCH = do_PUT

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (the base class's name)
        # The request line and status only: the key travels in a header, never here.
        sys.stderr.write(f"{datetime.now():%H:%M:%S} {format % args}\n")


def write_record(api: Api, kind: str, body: dict) -> str:
    """The record into the run folder, named by kind, then the set, variant
    and backend it gives (when they are plain names), then a sequence number;
    with the API's git commit, the models' hashes and the time added."""
    from bestbefore import runs

    labels = [str(body.get(k)) for k in ("set", "variant", "backend")
              if isinstance(body.get(k), str) and _LABEL.fullmatch(body[k])]
    name = "-".join([kind, *labels, f"{api.next_seq():03d}"]) + ".json"
    record = {**body, "kind": kind, "received": datetime.now().isoformat(timespec="seconds"),
              "api": {"run": api.run_dir.name, **runs.git_state()},
              "models": {v: m["sha256"] for v, m in api.models.items()}}
    runs.write_json(api.run_dir / name, record)
    return name


def make_server(api: Api, port: int = PORT) -> ThreadingHTTPServer:
    """The server on 127.0.0.1 only, never on another address."""
    handler = type("LiveHandler", (Handler,), {"api": api})
    return ThreadingHTTPServer((HOST, port), handler)


def make_api(with_store: bool, run_name: str = "m2c-app") -> Api:
    """The model files (each checked against the manifest first), the sets
    (v1-val only with --with-store), M2b's threshold, and a run folder."""
    from bestbefore import live, runs, sets
    from bestbefore.yolo import CLASSES

    folder = live.models_dir()
    manifest = live.read_manifest(folder)
    variants = [v for v in live.FILES if v in manifest["files"] and (folder / live.FILES[v]).exists()]
    if problems := live.check_files(folder, variants):
        sys.exit("\n".join(problems))
    models = {v: {"path": folder / live.FILES[v], "sha256": manifest["files"][v]["sha256"],
                  "bytes": manifest["files"][v]["bytes"]} for v in variants}
    served = PUBLIC_SETS + (STORE_SETS if with_store else ())
    photo_sets = {name: [Photo(s.width, s.height, Path(s.image)) for s in sets.load_set(name)] for name in served}
    config = {"threshold": live.threshold(), "classes": list(CLASSES), "imgsz": live.IMGSZ}
    run_dir = runs.start_run(run_name, {**config, "with_store": with_store, "sets": list(served),
                                        "manifest": manifest}, root=runs.RUNS_DIR / "m2c")
    return Api(key=secrets.token_urlsafe(24), models=models, sets=photo_sets,
               refused=() if with_store else STORE_SETS, run_dir=run_dir, config=config)


def lan_address() -> str | None:
    """This laptop's address on the Wi-Fi (a UDP socket's own address once
    pointed at a private one; nothing is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-store", action="store_true",
                   help="also serve v1-val (store photos): for the laptop's own browser only")
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()
    api = make_api(args.with_store)
    server = make_server(api, args.port)
    print(f"API on http://{HOST}:{args.port}, run folder {api.run_dir}", flush=True)
    print(f"sets: {', '.join(api.sets)}; models: {', '.join(api.models)}", flush=True)
    print(f"laptop: https://localhost:{WEB_PORT}/#k={api.key}", flush=True)
    if (ip := lan_address()) and not args.with_store:
        print(f"phone:  https://{ip}:{WEB_PORT}/#k={api.key}", flush=True)
    elif args.with_store:
        print("store photos are served: open the app on this laptop only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
