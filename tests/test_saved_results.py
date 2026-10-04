"""Validate a completed sweep CSV against its metadata."""

import csv
import json
import math
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("MCSK_RESULTS_DIR", ROOT / "results"))


class SavedSweepTests(unittest.TestCase):
    def test_saved_sweep_is_complete_and_matches_current_decoder(self):
        csv_path = RESULTS / "sweep_results.csv"
        metadata_path = RESULTS / "sweep_metadata.json"
        if not csv_path.is_file() or not metadata_path.is_file():
            self.skipTest("Run a sweep first to create results/sweep_results.csv and metadata")

        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata.get("scoring_method"),
                         "shared valley-minimum baseline",
                         "Results predate the current decoder; rerun the sweep")
        schemes = metadata["schemes"]
        self.assertEqual(set(schemes), {"permutation", "combo_fixed", "sparse"})
        expected_total = metadata["pooled_symbols_per_point"]

        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            columns = set(reader.fieldnames or ())

        required = {
            "sweep", "scheme", "num_molecule_types", "n_tx", "ts", "errors",
            "total_symbols", "ser", "ser_ci_low", "ser_ci_high", "ber_approx",
            "bits_per_symbol", "goodput_bits_per_second",
        }
        self.assertTrue(required.issubset(columns))
        expected_rows = len(schemes) * len(metadata["molecule_type_counts"]) * (
            len(metadata["ts_values_seconds"]) + len(metadata["n_tx_values"]))
        self.assertEqual(len(rows), expected_rows)

        for row in rows:
            self.assertEqual(int(row["total_symbols"]), expected_total)
            self.assertLessEqual(int(row["errors"]), int(row["total_symbols"]))
            for column in required - {"sweep", "scheme"}:
                self.assertTrue(math.isfinite(float(row[column])),
                                f"Non-finite {column} in {row}")

        # Report separated adjacent confidence intervals for review; these
        # are observations, not automatic failures.
        groups = {}
        for row in rows:
            group = (row["scheme"], row["num_molecule_types"], row["sweep"])
            groups.setdefault(group, []).append(row)
        separated = []
        for (scheme, k, sweep_kind), group_rows in groups.items():
            x_name = "ts" if sweep_kind == "ts" else "n_tx"
            ordered = sorted(group_rows, key=lambda row: float(row[x_name]))
            for lower, higher in zip(ordered, ordered[1:]):
                if float(higher["ser_ci_low"]) > float(lower["ser_ci_high"]):
                    separated.append((scheme, k, sweep_kind,
                                      lower[x_name], higher[x_name]))
        print("Adjacent points with non-overlapping Wilson intervals:", separated)


if __name__ == "__main__":
    unittest.main()
