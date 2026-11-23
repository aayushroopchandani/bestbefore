"""M2c S1's model files (docs/m2c-live-boxes.md, "The model files"): the
export, float16, int8 with the head's decoding left in float, the manifest,
the calibration photos, ONNX Runtime's threads, and rect reaching Ultralytics.

The export tests use a tiny untrained YOLO11n built from Ultralytics' own
yaml (no download, 64 px), exported by the same functions as the M2b
detector. They need onnxslim (the live extra), so they skip on the cluster,
which scores the files but never exports them."""
import ast
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bestbefore import live, m2b

TINY = 64


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """A tiny YOLO11n's files: fp32 exported, fp16, fp16-head32, int8 and
    int8-pct made from it (calibrated on two made-up photos), with a manifest."""
    pytest.importorskip("onnxslim")
    import onnx
    import torch
    from ultralytics import YOLO, utils

    folder = tmp_path_factory.mktemp("m2c")
    torch_model = YOLO("yolo11n.yaml")
    # Untrained, its head's outputs are all but constant (every class sigmoid
    # 0.5, every DFL logit 2.0, within 1e-6), too narrow for the Percentile
    # calibrator's 2,048 histogram bins. Seeded noise makes them vary, as a
    # trained model's do.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in torch_model.model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    torch_model.save(str(folder / "tiny.pt"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(utils, "AUTOINSTALL", False)      # onnxslim is installed: nothing would be
        fp32 = live.export_fp32(folder / "tiny.pt", folder / live.FILES["fp32"], imgsz=TINY)
    model = onnx.load(str(fp32))
    nodes = live.decode_nodes(model)
    onnx.save(live.to_float16(model), str(folder / live.FILES["fp16"]))
    onnx.save(live.to_float16(model, nodes), str(folder / live.FILES["fp16-head32"]))
    photos = []
    for i, size in enumerate([(160, 60), (60, 160)]):       # wide and tall: tight differs from square
        rng = np.random.default_rng(i)
        photos.append(folder / f"photo{i}.jpg")
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), np.uint8)).save(photos[-1])
    readers = {}
    for variant, method in (("int8", "minmax"), ("int8-pct", "percentile")):
        readers[variant] = live.CalibrationReader(photos, imgsz=TINY)
        live.quantize_int8(fp32, folder / live.FILES[variant], readers[variant], method, nodes)
    live.update_manifest(folder, {v: live.file_entry(folder / live.FILES[v]) for v in live.FILES})
    return SimpleNamespace(folder=folder, fp32=fp32, model=model, nodes=nodes, readers=readers,
                           last_module=len(torch_model.model.model) - 1)


def _load(path):
    import onnx
    return onnx.load(str(path))


def test_the_source_is_the_m2b_detector():
    assert live.SOURCE == "n1280-ft-s0"
    assert m2b.LAST_PT_SHA256[live.SOURCE].startswith("c839af8a")


def test_the_manifest_hashes_match_the_files(tiny, tmp_path):
    assert live.check_files(tiny.folder, list(live.FILES)) == []
    copy = tmp_path / "copy"
    shutil.copytree(tiny.folder, copy)
    with open(copy / live.FILES["fp16"], "ab") as f:
        f.write(b"\0")
    (copy / live.FILES["int8"]).unlink()
    problems = live.check_files(copy, list(live.FILES))
    assert len(problems) == 2
    assert problems[0].startswith("fp16:") and problems[1].startswith("int8:")
    manifest = json.loads((tiny.folder / live.MANIFEST).read_text())
    assert manifest["updates"][-1]["versions"]["onnxruntime"]
    assert manifest["files"]["fp32"]["opset"]["ai.onnx"] >= 17


@pytest.mark.parametrize("variant", ["fp16", "fp16-head32", "int8", "int8-pct"])
def test_derived_files_keep_float32_inputs_outputs_dynamic_axes_and_metadata(tiny, variant):
    import onnxruntime

    path = tiny.folder / live.FILES[variant]
    assert live.structure_problems(path, tiny.fp32) == []
    entry = live.file_entry(path)
    assert entry["inputs"] == [{"name": "images", "type": "FLOAT", "shape": ["batch", 3, "height", "width"]}]
    assert entry["outputs"][0]["type"] == "FLOAT" and entry["outputs"][0]["shape"][::2] == ["batch", "anchors"]
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    fp32 = {p.key: p.value for p in tiny.model.metadata_props}
    assert meta == fp32 and len(ast.literal_eval(meta["names"])) == 80 and meta["stride"] == "32"
    out = session.run(None, {"images": np.zeros((1, 3, TINY, 96), np.float32)})[0]
    assert out.shape == (1, 84, 8 * 12 + 4 * 6 + 2 * 3)       # a tight 64 x 96 input: 126 anchors


def test_the_decoding_is_the_last_modules_nodes_outside_its_convolutions(tiny):
    graph, nodes = tiny.model.graph, tiny.nodes
    by_name = {n.name: n for n in graph.node}
    head = f"/model.{tiny.last_module}/"
    assert live.head_module(tiny.model) == tiny.last_module          # read from the graph, not written in
    assert nodes and all(n.startswith(head) for n in nodes)
    assert not any(n.startswith((head + "cv2.", head + "cv3.")) for n in nodes)
    ops = {by_name[n].op_type for n in nodes}
    assert {"Sigmoid", "Softmax", "Concat", "Range"} <= ops
    assert [n for n in nodes if by_name[n].op_type == "Conv"] == [head + "dfl/conv/Conv"]
    writer = next(n.name for n in graph.node if "output0" in n.output)
    assert writer in nodes
    assert live.decode_problems(tiny.model, nodes) == []


def test_a_wrong_decoding_list_is_refused(tiny):
    graph, nodes = tiny.model.graph, tiny.nodes
    sigmoid = next(n for n in nodes if n.endswith("/Sigmoid"))
    without = [n for n in nodes if n != sigmoid]
    assert any("reads the head's convolutions" in p for p in live.decode_problems(tiny.model, without))
    assert any("no Sigmoid" in p for p in live.decode_problems(tiny.model, without))
    branch_conv = next(n.name for n in graph.node if "/cv2.0/" in n.name and n.op_type == "Conv")
    assert any("a convolution outside the DFL" in p for p in live.decode_problems(tiny.model, nodes + [branch_conv]))
    backbone = next(n.name for n in graph.node if n.name.startswith("/model.0/"))
    assert any("not in the last module" in p for p in live.decode_problems(tiny.model, nodes + [backbone]))
    assert live.decode_problems(tiny.model, []) == ["no node"]


@pytest.mark.parametrize("variant", ["int8", "int8-pct"])
def test_int8_quantises_the_convolutions_but_never_the_decoding(tiny, variant):
    q = _load(tiny.folder / live.FILES[variant])
    by_name = {n.name: n for n in q.graph.node}
    for name in tiny.nodes:            # every decoding node still there, as it was
        assert by_name[name].op_type == next(n.op_type for n in tiny.model.graph.node if n.name == name)
    decoded = {t for name in tiny.nodes for t in by_name[name].output}
    assert not [n.name for n in q.graph.node if n.op_type == "QuantizeLinear" and n.input[0] in decoded]
    writers = {t: n for n in q.graph.node for t in n.output}
    dfl = by_name[f"/model.{tiny.last_module}/dfl/conv/Conv"]
    assert all(writers.get(t) is None or writers[t].op_type != "DequantizeLinear" for t in dfl.input[1:])
    branch = [n for n in q.graph.node if n.op_type == "Conv" and "/cv3.0/" in n.name]
    assert branch and all(writers[n.input[1]].op_type == "DequantizeLinear" for n in branch)   # int8 weights
    assert sum(n.op_type == "QuantizeLinear" for n in q.graph.node) > 50


def test_int8_calibrates_on_every_photo_one_at_a_time_letterboxed_tight(tiny):
    for reader in tiny.readers.values():
        assert reader.shapes == [(1, 3, 32, 64), (1, 3, 64, 32)]      # 160x60 -> 64x24, padded to 32 tall


def test_fp16_head32_keeps_the_decoding_float32(tiny):
    head32, plain = _load(tiny.folder / live.FILES["fp16-head32"]), _load(tiny.folder / live.FILES["fp16"])
    assert live.float16_problems(head32, tiny.nodes, tiny.model) == []
    assert any("reads or writes float16" in p for p in live.float16_problems(plain, tiny.nodes, tiny.model))
    assert live.float16_problems(plain, (), tiny.model) == []
    names = [n.name for n in head32.graph.node]
    assert len(names) == len(set(names))                               # the converter alone repeats Cast names


def test_the_letterbox_is_ultralytics_tight_or_square():
    image = Image.new("RGB", (160, 90), (255, 255, 255))
    tight, square = live.letterboxed(image, 64), live.letterboxed(image, 64, tight=False)
    assert tight.shape == (1, 3, 64, 64) and square.shape == (1, 3, 64, 64)   # 36 rows pad to 64 either way
    tall = live.letterboxed(Image.new("RGB", (90, 300), (255, 255, 255)), 64)
    assert tall.shape == (1, 3, 64, 32) and tall.dtype == np.float32           # 19 columns -> a multiple of 32
    assert np.isclose(tall.max(), 1.0) and np.isclose(tall[0, :, :, 0].mean(), 114 / 255)   # grey 114 at the edge


def _fake_store_export(root, photos):
    """A store export as store_train writes one: per photo its turns and part."""
    folders = {"train": {"train_0": 0, "train_90": 1}, "val": {"val": 0}}
    lists = {"train": [], "val": []}
    for stem, part in photos.items():
        for folder in folders[part]:
            path = root / "images" / folder / f"{stem}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (120, 80), (90, 90, 90)).save(path)
            lists[part].append(f"./images/{folder}/{stem}.jpg\n")
    for part, lines in lists.items():
        (root / f"{part}.txt").write_text("".join(lines))
    (root / "manifest.json").write_text(json.dumps({
        "settings": {"folders": folders, "lists": {"train": "train.txt", "val": "val.txt"}},
        "photos": {stem: {"part": part} for stem, part in photos.items()}}))


def test_calibration_takes_training_photos_only_upright_by_seed(fake_expdate, tmp_path, monkeypatch):
    store = tmp_path / "store"
    _fake_store_export(store, {"v1-0001": "train", "v3-0002": "train", "v1-0003": "val"})
    monkeypatch.setenv("STORE_TRAIN_DIR", str(store))
    photos = live.calibration_photos(store=2, expdate=2)
    ids = [i for i, _ in photos]
    assert sorted(ids[:2]) == ["store:v1-0001", "store:v3-0002"]
    assert sorted(ids[2:]) == ["expdate:a.jpg", "expdate:b.jpg"]          # train_real: never v1/v2/v3.jpg
    assert all(p.parent.name == "train_0" for i, p in photos if i.startswith("store:"))   # upright only
    assert live.calibration_photos(store=1, expdate=1) == [photos[0], photos[2]]   # a smaller draw starts the same
    assert live.list_sha256(photos) == live.list_sha256(live.calibration_photos(2, 2))
    with pytest.raises(ValueError, match="there are 2 and 2"):
        live.calibration_photos(store=3, expdate=1)


def test_calibration_refuses_a_validation_or_v2_photo(fake_expdate, tmp_path, monkeypatch):
    store = tmp_path / "store"
    _fake_store_export(store, {"v1-0001": "train", "v2-0004": "train"})
    monkeypatch.setenv("STORE_TRAIN_DIR", str(store))
    with pytest.raises(ValueError, match="store set v2"):
        live.calibration_photos(store=2, expdate=0)
    manifest = json.loads((store / "manifest.json").read_text())
    manifest["photos"]["v2-0004"]["part"] = "val"           # listed for training, marked validation
    (store / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="isn't a training photo"):
        live.calibration_photos(store=2, expdate=0)
    yolo = fake_expdate.parent / "expdate" / "yolo"
    (yolo / "train_real.txt").write_text((yolo / "train_real.txt").read_text() + (yolo / "val.txt").read_text())
    with pytest.raises(ValueError, match="is a validation photo"):
        live.calibration_photos(store=0, expdate=5)


def test_the_calibration_reader_strides_and_rewinds(tmp_path):
    paths = []
    for i in range(3):
        paths.append(tmp_path / f"{i}.jpg")
        Image.new("RGB", (100, 50), (i * 60, 0, 0)).save(paths[-1])
    reader = live.CalibrationReader(paths, imgsz=TINY)
    assert len(reader) == 3
    reader.set_range(1, 2)
    first = reader.get_next()["images"]
    assert reader.get_next() is None
    reader.rewind()
    assert np.array_equal(reader.get_next()["images"], first)
    assert first.shape == (1, 3, 32, 64) and 0 <= first.min() and first.max() <= 1


def test_every_onnx_runtime_session_made_inside_runs_the_threads_asked():
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    data = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8).SerializeToString()
    original = onnxruntime.InferenceSession
    with live.ort_threads(3) as made:
        plain = onnxruntime.InferenceSession(data, providers=["CPUExecutionProvider"])
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        given = onnxruntime.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    assert made == [3, 3]
    assert [s.get_session_options().intra_op_num_threads for s in (plain, given)] == [3, 3]
    assert given.get_session_options().graph_optimization_level == onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert onnxruntime.InferenceSession is original


def test_the_export_refuses_to_overwrite_and_to_autoinstall(tmp_path, monkeypatch):
    from ultralytics import utils

    (tmp_path / "det-fp32.onnx").write_bytes(b"made before")
    with pytest.raises(FileExistsError):
        live.export_fp32(tmp_path / "x.pt", tmp_path / "det-fp32.onnx")
    monkeypatch.setattr(utils, "AUTOINSTALL", True)
    with pytest.raises(RuntimeError, match="YOLO_AUTOINSTALL"):
        live.export_fp32(tmp_path / "x.pt", tmp_path / "other.onnx")
