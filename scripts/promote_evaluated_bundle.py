"""Activate the frozen evaluated bundle while preserving the integration build."""

import hashlib
import json
import shutil
from pathlib import Path

from credit_score.service import load_service, request_for


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    active = root / "artifacts"
    evaluated = active / "stage-two"
    service = load_service(evaluated)
    meta = service.metadata
    report_path = root / meta["evaluation_report"]
    report = json.loads(report_path.read_text())
    assert digest(report_path) == meta["evaluation_report_sha256"]
    assert report["model_sha256"] == digest(evaluated / "pipeline.joblib")
    assert digest(active / "split-indices.npz") == meta["split_indices_sha256"]
    service.predict(request_for(meta["defaults"]))
    old = json.loads((active / "metadata.json").read_text())
    if old["model_version"] != meta["model_version"]:
        if not old["model_version"].startswith("stage1-"):
            raise RuntimeError("Unexpected active version; inspect before replacing it")
        checkpoint = root / "qa/checkpoints" / old["model_version"]
        checkpoint.mkdir(parents=True, exist_ok=True)
        for name in ["pipeline.joblib", "metadata.json", "split-indices.npz"]:
            target = checkpoint / name
            if target.exists():
                assert digest(target) == digest(active / name)
            else:
                shutil.copy2(active / name, target)
        # The app reloads on metadata mtime, after the model file is in place.
        shutil.copyfile(evaluated / "pipeline.joblib", active / "pipeline.joblib")
        shutil.copyfile(evaluated / "metadata.json", active / "metadata.json")
    assert load_service(active).metadata["model_version"] == meta["model_version"]
    payload = request_for(meta["defaults"])
    for name, value in [
        ("request", payload),
        ("response", load_service(active).predict(payload)),
    ]:
        (root / "examples" / (name + ".json")).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf8"
        )
    print(f"Active local bundle: {meta['model_version']}; examples regenerated")


if __name__ == "__main__":
    main()
