"""M2c S2: the API the live-boxes app talks to (scripts/live_api.py;
docs/m2c-live-boxes.md, "Serving it to the phone"), served on a free port
over made-up photos and a made-up model file."""
import hashlib
import http.client
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_api  # noqa: E402

from bestbefore import runs  # noqa: E402

KEY = "test-key"


@pytest.fixture
def serve(tmp_path):
    """Starts the API over two made-up sets; yields a request function."""
    servers = []

    def start(with_store: bool):
        model = tmp_path / "det-fp32.onnx"
        model.write_bytes(b"not really a model")
        photos = []
        for i, (w, h) in enumerate([(40, 30), (30, 50)]):
            path = tmp_path / f"photo{i}.jpg"
            Image.fromarray(np.random.default_rng(i).integers(0, 255, (h, w, 3), np.uint8)).save(path, quality=90)
            photos.append(live_api.Photo(w, h, path))
        api = live_api.Api(
            key=KEY, models={"fp32": {"path": model, "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                                      "bytes": model.stat().st_size}},
            sets={"real-val": photos, **({"v1-val": photos[:1]} if with_store else {})},
            refused=() if with_store else live_api.STORE_SETS, run_dir=tmp_path / "run",
            config={"threshold": 0.1640664, "classes": ["date", "due", "prod", "code"], "imgsz": 1280})
        api.run_dir.mkdir()
        server = live_api.make_server(api, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)

        def request(method, path, key=KEY, body=None, headers=None):
            conn = http.client.HTTPConnection(*server.server_address[:2], timeout=10)
            h = {**({live_api.KEY_HEADER: key} if key else {}), **(headers or {})}
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            data = r.read()
            conn.close()
            return r.status, dict(r.getheaders()), data

        return api, server, request

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def test_it_listens_on_127_0_0_1_only(serve):
    _, server, _ = serve(False)
    assert server.server_address[0] == "127.0.0.1"


def test_every_route_needs_the_key(serve):
    api, _, request = serve(False)
    sha = api.models["fp32"]["sha256"]
    for method, path in [("GET", "/api/config"), ("GET", f"/api/models/fp32/{sha}"), ("GET", "/api/bench/real-val"),
                         ("GET", "/api/bench/real-val/0"), ("GET", "/api/bench/real-val/0/input"),
                         ("POST", "/api/records/live"), ("GET", "/anything")]:
        for key in (None, "wrong", KEY[:-1], KEY + "x"):
            status, _, _ = request(method, path, key=key, body=b"{}")
            assert status == 401, (method, path, key)
    assert request("GET", "/api/config")[0] == 200


def test_no_path_reaches_out_of_its_folder(serve):
    api, _, request = serve(False)
    sha = api.models["fp32"]["sha256"]
    for path in ["/api/models/../../.env", f"/api/models/fp32/{sha}/../../x", "/api/models/%2e%2e/%2e%2e/.env",
                 "/api/bench/real-val/../../../../etc/passwd", "/api/bench/..%2f..%2fdata/0", "/api/bench/real-val/-1",
                 "/api/bench/real-val/2", "/api/models/fp16/" + sha, "/api/models/fp32/" + "0" * 64,
                 "/api/models/fp32/" + sha.upper(), "/api/config/", "/api/records/../x"]:
        status, _, _ = request("GET", path)
        assert status == 404, path
    assert request("POST", "/api/records/../../x", body=b"{}")[0] == 404
    assert request("POST", "/api/records/notes", body=b"{}")[0] == 404
    assert request("POST", "/api/records/error", body=b"{\"where\": \"bench\"}")[0] == 201      # what failed on the phone
    assert request("DELETE", "/api/config")[0] == 405


def test_the_model_is_served_by_its_hash_and_cached(serve):
    api, _, request = serve(False)
    sha = api.models["fp32"]["sha256"]
    status, headers, data = request("GET", f"/api/models/fp32/{sha}")
    assert status == 200 and data == b"not really a model"
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["Cache-Control"] == "private, max-age=31536000, immutable" and headers["ETag"] == f'"{sha}"'
    config = json.loads(request("GET", "/api/config")[2])
    assert config["models"] == {"fp32": {"sha256": sha, "bytes": 18, "url": f"/api/models/fp32/{sha}"}}
    assert config["threshold"] == 0.1640664 and config["sets"] == ["real-val"]


def test_photos_go_by_set_and_index_with_their_decoded_hash(serve):
    api, _, request = serve(False)
    listing = json.loads(request("GET", "/api/bench/real-val")[2])
    assert listing == {"set": "real-val", "photos": [{"index": 0, "width": 40, "height": 30},
                                                      {"index": 1, "width": 30, "height": 50}]}
    assert "photo0" not in json.dumps(listing)
    status, headers, data = request("GET", "/api/bench/real-val/1")
    path = api.sets["real-val"][1].path
    assert status == 200 and headers["Content-Type"] == "image/jpeg" and data == path.read_bytes()
    pixels = np.asarray(Image.open(path).convert("RGB"))
    assert headers["X-Decoded-Sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()


def test_the_input_is_ultralytics_letterbox(serve):
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    api, _, request = serve(False)
    status, headers, data = request("GET", "/api/bench/real-val/0/input")
    assert status == 200 and (headers["X-Input-Width"], headers["X-Input-Height"]) == ("1280", "1280")
    image = np.asarray(Image.open(api.sets["real-val"][0].path).convert("RGB"))
    expected = LetterBox((1280, 1280), auto=False, stride=32)(image=image)
    assert data == expected.tobytes() and len(data) == 1280 * 1280 * 3
    assert expected[0, 0].tolist() == [114, 114, 114]          # 40 x 30 scaled to 1280 x 960: grey above


def test_v1_val_is_refused_without_with_store(serve):
    _, _, request = serve(False)
    for path in ["/api/bench/v1-val", "/api/bench/v1-val/0", "/api/bench/v1-val/0/input"]:
        status, _, data = request("GET", path)
        assert status == 403 and b"--with-store" in data, path
    assert "v1-val" not in json.loads(request("GET", "/api/config")[2])["sets"]


def test_with_store_v1_val_goes_to_the_laptops_browser_only(serve):
    _, _, request = serve(True)
    laptop = {"X-Forwarded-For": "127.0.0.1"}
    assert request("GET", "/api/bench/v1-val/0", headers=laptop)[0] == 200
    assert request("GET", "/api/bench/v1-val/0", headers={"X-Forwarded-For": "::1"})[0] == 200
    assert request("GET", "/api/bench/v1-val")[0] == 200                   # no proxy: the laptop itself
    for phone in ["192.168.1.23", "127.0.0.1, 192.168.1.23", "::ffff:10.0.0.4", "not-an-address"]:
        for path in ["/api/bench/v1-val", "/api/bench/v1-val/0", "/api/bench/v1-val/0/input"]:
            assert request("GET", path, headers={"X-Forwarded-For": phone})[0] == 403, (phone, path)
        sets = json.loads(request("GET", "/api/config", headers={"X-Forwarded-For": phone})[2])["sets"]
        assert sets == ["real-val"]
        assert request("GET", "/api/bench/real-val/0", headers={"X-Forwarded-For": phone})[0] == 200
    assert json.loads(request("GET", "/api/config", headers=laptop)[2])["sets"] == ["real-val", "v1-val"]


def test_a_record_is_written_with_the_commit_the_models_and_the_time(serve):
    api, _, request = serve(False)
    body = {"set": "real-val", "variant": "fp32", "backend": "webgpu", "photos": [{"index": 0, "boxes": []}]}
    status, _, data = request("POST", "/api/records/bench-parity", body=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
    assert status == 201
    name = json.loads(data)["file"]
    assert name == "bench-parity-real-val-fp32-webgpu-001.json"
    record = json.loads((api.run_dir / name).read_text())
    assert record["photos"] == body["photos"] and record["kind"] == "bench-parity" and record["received"]
    assert record["api"]["git_commit"] == runs.git_state()["git_commit"] and record["api"]["run"] == "run"
    assert record["models"] == {"fp32": api.models["fp32"]["sha256"]}
    # Names come from plain labels only, and never repeat.
    odd = {"set": "../../x", "variant": "FP32", "backend": "webgpu"}
    status, _, data = request("POST", "/api/records/live", body=json.dumps(odd).encode())
    assert status == 201 and json.loads(data)["file"] == "live-webgpu-002.json"
    assert sorted(p.name for p in api.run_dir.iterdir()) == ["bench-parity-real-val-fp32-webgpu-001.json",
                                                             "live-webgpu-002.json"]


def test_a_record_must_be_a_json_object(serve):
    _, _, request = serve(False)
    assert request("POST", "/api/records/live", body=b"not json")[0] == 400
    assert request("POST", "/api/records/live", body=b"[1, 2]")[0] == 400
    assert request("POST", "/api/records/live", body=b"")[0] == 411


def test_every_answer_has_the_safety_headers(serve):
    api, _, request = serve(False)
    for path in ["/api/config", "/api/bench/real-val", "/api/bench/real-val/0", "/api/nothing"]:
        _, headers, _ = request("GET", path)
        assert headers["X-Content-Type-Options"] == "nosniff", path
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin", path
        assert headers["Referrer-Policy"] == "no-referrer", path
    _, headers, _ = request("GET", "/api/config")
    assert headers["Cache-Control"] == "no-store" and headers["Content-Type"] == "application/json"
    _, headers, _ = request("GET", "/api/config", key=None)
    assert headers["Cache-Control"] == "no-store"
