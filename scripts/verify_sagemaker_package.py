"""Verify packaged inference in a fresh process, not an AWS deployment."""

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from credit_score.serving_policy import load_exploration_service as load_service


def main():
    root = Path(__file__).resolve().parents[1]
    service = load_service(root / "artifacts")
    defaults = service.metadata["defaults"]
    payload = {
        "contract_version": "1.0",
        "instances": [
            defaults,
            {**defaults, "Age": 70, "Annual_Income": 120000000},
            {**defaults, "Monthly_Inhand_Salary": None},
        ],
    }
    expected = service.predict(payload)
    with tempfile.TemporaryDirectory(
        prefix="sagemaker-package-", dir=root / "qa"
    ) as temporary:
        folder = Path(temporary)
        with tarfile.open(root / "qa/sagemaker/model.tar.gz") as archive:
            names = archive.getnames()
            assert not any(x.endswith((".csv", ".npz")) for x in names)
            archive.extractall(folder, filter="data")
        code = """import inference,json,sys
folder=sys.argv[1]
payload=json.loads(sys.stdin.read())
service=inference.model_fn(folder)
request=inference.input_fn(json.dumps(payload),'application/json')
response,kind=inference.output_fn(inference.predict_fn(request,service),'application/json')
print(response)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(folder)],
            cwd=folder / "code",
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        actual = json.loads(result.stdout)
        assert actual == expected
    record = {
        "scope": "Fresh-process packaged hooks; no AWS resource called",
        "model_version": service.metadata["model_version"],
        "local_matches_packaged": True,
        "dataset_absent": True,
        "split_indices_absent": True,
        "profiles_checked": 3,
        "outside_range_warning_parity": True,
        "members": names,
    }
    target = root / "docs/verification"
    target.mkdir(parents=True, exist_ok=True)
    (target / "sagemaker-package.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
