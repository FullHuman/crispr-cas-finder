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

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            finder.find_crispr_arrays(">tiny\nACGT", min_repeat_length=0)
        with self.assertRaises(ValueError):
            finder.find_crispr_arrays(">tiny\nACGT", min_evidence_level=5)


if __name__ == "__main__":
    unittest.main()
