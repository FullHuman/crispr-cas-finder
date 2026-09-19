import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("bundle_cas_models", Path(__file__).resolve().parents[1] / "bundle_cas_models.py")
bundler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundler)


class BundleTests(unittest.TestCase):
    def test_repository_models_are_complete(self):
        bundle = bundler.build_bundle(bundler.DEFAULT_CAS_DIR)
        self.assertEqual(len(bundle["models"]), 23)
        self.assertEqual(len(bundle["profiles"]), 121)
        self.assertEqual(len({p["name"] for p in bundle["profiles"]}), 121)

    def test_missing_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No CAS models"):
                bundler.build_bundle(Path(directory))


if __name__ == "__main__":
    unittest.main()
