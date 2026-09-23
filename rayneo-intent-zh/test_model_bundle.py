"""Small fixture for the same split, checksum, and atomic restore path as v5."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from model_bundle import RUNTIME_FILES, package_model, restore_model


class ModelBundleTest(unittest.TestCase):
    def test_reconstructs_and_rejects_a_corrupt_lfs_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, restored = root / "source", root / "bundle", root / "restored"
            source.mkdir()
            weights = bytes(range(101)) * 3
            (source / "best.safetensors").write_bytes(weights)
            for name in RUNTIME_FILES:
                item = source / name
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_bytes(name.encode())
            digest = hashlib.sha256(weights).hexdigest()
            manifest = package_model(source, bundle, digest, part_bytes=67)
            self.assertGreater(len(manifest["parts"]), 1)
            part = bundle / manifest["parts"][1]["name"]
            original = part.read_bytes()
            part.write_bytes(b"X" + original[1:])
            with self.assertRaisesRegex(ValueError, "corrupt"):
                restore_model(bundle, restored, digest)
            self.assertFalse((restored / "best.safetensors").exists())
            part.write_bytes(original)
            restore_model(bundle, restored, digest)
            self.assertEqual((restored / "best.safetensors").read_bytes(), weights)
            restore_model(bundle, restored, digest)  # A second deployment is idempotent.


if __name__ == "__main__":
    unittest.main()
