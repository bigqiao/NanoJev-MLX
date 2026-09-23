#!/usr/bin/env python3
"""Verify that head export preserved every non-scalar tensor exactly."""
import argparse
import json
from pathlib import Path
from safetensors import safe_open
from evaluate import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, candidate = args.base / "best.safetensors", args.candidate / "best.safetensors"
    changed = []
    with safe_open(str(base), framework="np") as old, safe_open(str(candidate), framework="np") as new:
        if set(old.keys()) != set(new.keys()):
            raise ValueError("Tensor keys differ.")
        count = len(old.keys())
        for name in old.keys():
            left, right = old.get_tensor(name), new.get_tensor(name)
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"Shape/dtype differs: {name}")
            if left.tobytes() != right.tobytes():
                changed.append(name)
    if set(changed) != {"scalar.weight", "scalar.bias"}:
        raise ValueError(f"Unexpected changed tensors: {changed}")
    result = {"base": args.base.name, "candidate": args.candidate.name,
              "baseWeightsSha256": digest(base), "candidateWeightsSha256": digest(candidate),
              "tensorCount": count, "changedTensors": sorted(changed), "nonScalarTensorsByteIdentical": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
