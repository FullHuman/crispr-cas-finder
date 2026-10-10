"""Public binding smoke tests; run against an installed/built wheel."""
from pathlib import Path
import tempfile
import unittest
import crispr_cas_finder as finder


class ApiTests(unittest.TestCase):
    def test_file_and_string_api_agree(self):
        dr = "GTTCACTGCCGTATAGGCAGCTAAGAAA"
        sequence = "TGCA" * 25 + dr + "A" * 32 + dr + "C" * 32 + dr + "TGCA" * 25
        fasta = ">synthetic\n" + sequence + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.fa"
            path.write_text(fasta)
            by_path = finder.find_crispr_arrays_in_file(str(path))
        by_text = finder.find_crispr_arrays(fasta)
        self.assertTrue(by_text)
        self.assertEqual([(a.seq_id, a.start, a.end, a.consensus_repeat) for a in by_path],
                         [(a.seq_id, a.start, a.end, a.consensus_repeat) for a in by_text])
        self.assertEqual(by_text[0].repeat_count, by_text[0].spacer_count + 1)

    def test_missing_file_and_malformed_fasta(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.fa"
            with self.assertRaises(FileNotFoundError):
                finder.find_crispr_arrays_in_file(str(path))
            path.write_text("not fasta\n")
            with self.assertRaises(ValueError):
                finder.find_crispr_arrays_in_file(str(path))

    def test_arrays_carry_ids_and_statistics(self):
        dr = "GTTCACTGCCGTATAGGCAGCTAAGAAA"
        sequence = "TGCA" * 25 + dr + "A" * 32 + dr + "C" * 32 + dr + "TGCA" * 25
        array = finder.find_crispr_arrays(">synthetic\n" + sequence + "\n")[0]
        self.assertEqual(array.id, "synthetic_CRISPR_1")
        self.assertEqual(array.statistics["exact_repeat_count"], array.repeat_count)
        self.assertEqual(array.statistics["mean_spacer_length"], 32.0)

    def test_analyze_returns_the_report_structure(self):
        dr = "GTTCACTGCCGTATAGGCAGCTAAGAAA"
        sequence = "TGCA" * 25 + dr + "A" * 32 + dr + "C" * 32 + dr + "TGCA" * 25
        report = finder.analyze(">synthetic plasmid\n" + sequence + "\n", locus_distance=1000)
        self.assertEqual(report["tool"]["name"], "crispr-cas-finder")
        self.assertEqual(report["sequences"][0]["description"], "plasmid")
        self.assertEqual(report["summary"]["crispr_count"], len(report["crisprs"]))
        self.assertEqual(report["crisprs"][0]["id"], "synthetic_CRISPR_1")
        self.assertEqual(report["cas_clusters"], [])
        self.assertIsNone(report["parameters"]["cas"])

    def test_analyze_requires_model_directories_for_cas(self):
        with self.assertRaises(ValueError):
            finder.analyze(">tiny\nACGT", cas=True)
        with tempfile.TemporaryDirectory() as directory:
            # A plain definitions directory also needs a profiles directory.
            with self.assertRaises(ValueError):
                finder.analyze(">tiny\nACGT", cas=True, cas_models_dir=directory)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            finder.find_crispr_arrays(">tiny\nACGT", min_repeat_length=0)
        with self.assertRaises(ValueError):
            finder.find_crispr_arrays(">tiny\nACGT", min_evidence_level=5)


if __name__ == "__main__":
    unittest.main()
