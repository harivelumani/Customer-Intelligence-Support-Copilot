from __future__ import annotations

import hashlib
import json
import sqlite3
import unittest
from pathlib import Path

import pandas as pd

from src.policy_agent import AgentRuntime, Retriever
from src.triage import _validate_response, redact

ROOT = Path(__file__).resolve().parents[1]


class ProjectAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = ROOT / "outputs"
        cls.part_a = json.loads((cls.output / "part_a_full_report.json").read_text(encoding="utf-8"))
        cls.part_b = json.loads((cls.output / "part_b_full_report.json").read_text(encoding="utf-8"))
        cls.part_c = json.loads((cls.output / "part_c_report.json").read_text(encoding="utf-8"))
        cls.part_d = json.loads((cls.output / "part_d_report.json").read_text(encoding="utf-8"))

    def test_protected_scripts_match_recovered_source_checksums(self) -> None:
        expected = {
            "generate_data.py": "943C9EC3E12CC68154FC20690CA787545781EE8B366311A6ABF23C65ABB6E77E",
            "mock_llm.py": "2A3038F496DF8643F759C61555E28E0ACE47F551AFD5E964477FAFD7EC03F6E0",
        }
        for filename, checksum in expected.items():
            digest = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest().upper()
            self.assertEqual(digest, checksum)

    def test_raw_audit_and_cleaning_checkpoints(self) -> None:
        audit = self.part_a["raw_audit"]
        self.assertEqual(audit["customers"]["shape"], [1200, 5])
        self.assertEqual(audit["products"]["shape"], [40, 4])
        self.assertEqual(audit["orders"]["shape"], [13084, 9])
        self.assertEqual(audit["checkpoints"], {
            "raw_order_rows": 13084,
            "exact_duplicate_order_rows": 119,
            "pending_orders": 656,
            "distinct_order_ids": 12309,
            "non_numeric_amounts": 120,
            "negative_delivery_times": 106,
            "blank_delivery_times": 2541,
            "blank_ages": 61,
            "ages_outside_10_100": 6,
            "raw_city_spellings": 24,
            "nonblank_raw_gender_spellings": 10,
            "blank_genders": 30,
        })
        clean = self.part_a["cleaning"]
        self.assertEqual(clean["orders"], 12309)
        self.assertEqual(clean["order_status_counts"], {"completed": 11110, "cancelled": 1199})
        self.assertEqual(clean["completed_revenue_inr"], 2998668.8)
        self.assertEqual(clean["completed_missing_delivery"], 0)
        self.assertEqual(clean["cancelled_missing_delivery"], 1199)
        self.assertEqual(clean["customer_missing_ages"], 0)

    def test_database_queries_and_pandas_crosscheck(self) -> None:
        analysis = self.part_a["analysis"]
        expected_rows = {"Q1": 30, "Q2": 30, "Q3": 11, "Q4": 25, "Q5": 550, "Q7": 183}
        for query, rows in expected_rows.items():
            self.assertEqual(analysis["sql"][query]["rows"], rows)
        q2 = pd.DataFrame(analysis["sql"]["Q2"]["records"])
        largest = q2.loc[q2["revenue_inr"].idxmax()]
        self.assertEqual((largest["city"], largest["category"]), ("Bengaluru", "Electronics"))
        self.assertAlmostEqual(largest["revenue_inr"], 233909.45, places=2)
        self.assertAlmostEqual(analysis["pivot_grand_total_inr"], 2998668.8, places=2)
        q6 = pd.DataFrame(analysis["sql"]["Q6"]["records"])
        self.assertEqual(int((q6["spend_rank"] != q6["city_row_number"]).sum()), 8)
        q7 = pd.DataFrame(analysis["sql"]["Q7"]["records"])
        self.assertAlmostEqual(q7["cumulative_revenue"].iloc[-1], 2998668.8, places=2)
        with sqlite3.connect(ROOT / "data" / "instamart.db") as connection:
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertTrue({"customers", "products", "orders"}.issubset(tables))

    def test_required_clean_csv_outputs(self) -> None:
        clean = ROOT / "data" / "clean"
        self.assertEqual(len(pd.read_csv(clean / "customers.csv")), 1200)
        self.assertEqual(len(pd.read_csv(clean / "products.csv")), 40)
        orders = pd.read_csv(clean / "orders.csv", keep_default_na=False)
        self.assertEqual(len(orders), 12309)
        self.assertEqual(orders["order_id"].nunique(), len(orders))

    def test_rfm_pca_lapse_model_and_retention_artifacts(self) -> None:
        clustering = self.part_b["clustering"]
        self.assertEqual(clustering["rfm_count"], 1175)
        self.assertAlmostEqual(clustering["rfm_summary"]["mean_monetary"], 2552.06, places=2)
        self.assertEqual(clustering["chosen_k"], 4)
        self.assertEqual(len(clustering["k_metrics"]), 7)
        self.assertAlmostEqual(sum(clustering["pca_explained_variance_ratio"]), 0.852, delta=0.02)
        model = self.part_b["lapse_prediction"]
        self.assertEqual((model["cohort_rows"], model["lapsed_rows"]), (1156, 443))
        self.assertEqual((model["train_rows"], model["test_rows"]), (924, 232))
        self.assertTrue(model["no_missing_features"])
        self.assertEqual(model["logistic_regression"]["balanced_train_class_counts"], {"0": 570, "1": 570})
        self.assertTrue(model["serialized_predictions_match"])
        self.assertGreaterEqual(
            max(model["logistic_regression"]["test"]["f1_lapsed"], model["random_forest"]["test"]["f1_lapsed"]),
            0.55,
        )
        for filename in ("customer_segments.csv", "lapse_model.joblib", "retention_list.csv"):
            self.assertTrue((self.output / filename).is_file())
        self.assertEqual(
            list(pd.read_csv(self.output / "customer_segments.csv").columns),
            ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"],
        )

    def test_ticket_redaction_validation_and_budget_outputs(self) -> None:
        redacted, counts = redact("Email x@y.com call 6789012345 order ORD-123456")
        self.assertEqual(counts, {"email": 1, "phone": 1, "order_id": 1})
        self.assertNotIn("x@y.com", redacted)
        self.assertEqual(_validate_response("ticket is billing-related")[0:2], ("classified_keyword", "billing"))
        self.assertEqual(_validate_response('{"category":"shipping","reason":"x"}')[0], "flagged_invalid_output")
        self.assertEqual(self.part_c["zero_shot"]["tickets_classified"], 147)
        self.assertAlmostEqual(self.part_c["zero_shot"]["accuracy"], 126 / 147)
        self.assertEqual(self.part_c["zero_shot"]["transient_exceptions"], 19)
        self.assertEqual(self.part_c["zero_shot"]["redaction_counts"], {"email": 55, "phone": 91, "order_id": 132})
        self.assertEqual(self.part_c["few_shot"]["accuracy"], 1.0)
        self.assertGreater(self.part_c["budget_demo"]["exhausted_tickets"], 0)
        self.assertEqual(self.part_c["final_few_shot"]["rows"], 164)
        for filename in ("triage_results.csv", "triage_budget_demo.csv"):
            frame = pd.read_csv(self.output / filename)
            self.assertEqual(len(frame), 164)
        triage = pd.read_csv(self.output / "triage_results.csv")
        self.assertEqual(
            list(triage.columns),
            ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"],
        )

    def test_policy_retrieval_scenarios_safety_and_resume(self) -> None:
        overlap = self.part_d["overlap_retrieval"]
        self.assertEqual(overlap["total_chunks"], 15)
        self.assertEqual(overlap["answerable_source_hits"], 12)
        self.assertEqual(overlap["answerable_fact_hits"], 12)
        self.assertEqual(overlap["unsupported_fallbacks"], 2)
        zero = self.part_d["zero_overlap_retrieval"]
        self.assertLess(zero["answerable_fact_hits"], 12)
        self.assertEqual(zero["unsupported_fallbacks"], 2)
        scenarios = self.part_d["scenario_results"]
        self.assertEqual([row["status"] for row in scenarios], [
            "REFUND", "REFUND", "ESCALATE", "DENIED", "DENIED", "NOT_FOUND",
            "FALLBACK_MAX_ITERATIONS",
        ])
        self.assertEqual(scenarios[0]["llm_calls"], 5)
        self.assertEqual(self.part_d["crash_recovery"]["tools_executed"], [
            "get_order", "search_policy", "calculate_refund", "issue_refund",
        ])
        self.assertEqual(self.part_d["token_limit_demo"]["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(self.part_d["timeout_demo"]["status"], "FALLBACK_TIMEOUT")
        runtime = AgentRuntime(Retriever(), "test")
        with self.assertRaises(ValueError):
            runtime.get_order("not-an-order")
        self.assertEqual(runtime.calculate_refund(250.0, 46)["refund_inr"], 250.0)
        self.assertEqual(runtime.calculate_refund(250.0, 45)["refund_inr"], 0.0)
        for filename in ("agent_runs.jsonl", "refund_ledger.jsonl"):
            path = self.output / filename
            self.assertTrue(path.is_file())
            if path.stat().st_size:
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(records)


if __name__ == "__main__":
    unittest.main()
