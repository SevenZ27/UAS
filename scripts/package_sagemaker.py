"""Package the verified pipeline and the same inference code, without raw data."""

import argparse
import tarfile
from pathlib import Path

from credit_score.service import load_service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, default=Path("qa/sagemaker/model.tar.gz")
    )
    args = parser.parse_args()
    load_service(args.artifacts)
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    requirements = args.output.parent / "requirements.txt"
    requirements.write_text(
        "scikit-learn==1.9.1\nnumpy==2.2.6\npandas==2.2.3\njoblib==1.6.0\n",
        encoding="utf8",
    )
    with tarfile.open(args.output, "w:gz") as archive:
        for name in ["pipeline.joblib", "metadata.json"]:
            archive.add(args.artifacts / name, arcname=name)
        archive.add(root / "inference.py", arcname="code/inference.py")
        archive.add(requirements, arcname="code/requirements.txt")
        for source in sorted((root / "credit_score").glob("*.py")):
            archive.add(source, arcname="code/credit_score/" + source.name)
    print(f"Packaged {args.output}; no dataset or split identifiers included.")


if __name__ == "__main__":
    main()
