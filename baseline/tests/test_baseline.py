import csv
import tempfile
import unittest
from pathlib import Path

from baseline import read_records, sari_score


class ReadRecordsTests(unittest.TestCase):
    def test_sentence_targets_include_split_text_and_empty_deletions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["pair_id", "complex", "simple"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "a",
                        "complex": "Complex sentence.",
                        "simple": "['Simple.', 'Next.']",
                    }
                )
                writer.writerow(
                    {"pair_id": "b", "complex": "Delete this.", "simple": "[]"}
                )
            records = read_records(path, "sentence")
        self.assertEqual(records[0]["reference"], "Simple. Next.")
        self.assertEqual(records[1]["reference"], "")

    def test_document_target_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["complex", "simple"])
                writer.writeheader()
                writer.writerow({"complex": "Clinical text.", "simple": "Plain text."})
            records = read_records(path, "document")
        self.assertEqual(records[0]["reference"], "Plain text.")


class SariTests(unittest.TestCase):
    def test_identical_source_prediction_and_reference_score(self):
        text = "the patient took medicine"
        self.assertAlmostEqual(sari_score(text, text, text), 100.0 / 3.0)

    def test_empty_prediction_can_match_deletion_reference(self):
        self.assertGreater(sari_score("remove this phrase", "", ""), 0)


if __name__ == "__main__":
    unittest.main()
