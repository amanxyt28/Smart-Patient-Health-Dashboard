import unittest
import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from health_utils import (
    build_prediction_frame,
    download_verified_artifact,
    filter_patient_records,
    paginate_frame,
)


class BuildPredictionFrameTests(unittest.TestCase):
    def setUp(self):
        self.input_values = {
            "heart_rate": 80.0,
            "spo2": 97.0,
            "temperature": 36.8,
        }

    def test_orders_columns_to_match_model_features(self):
        frame = build_prediction_frame(
            self.input_values,
            ["temperature", "heart_rate", "spo2"],
        )

        self.assertEqual(
            list(frame.columns), ["temperature", "heart_rate", "spo2"]
        )
        self.assertEqual(frame.iloc[0].tolist(), [36.8, 80.0, 97.0])

    def test_rejects_missing_model_feature(self):
        with self.assertRaisesRegex(ValueError, "Missing input values"):
            build_prediction_frame(self.input_values, ["blood_pressure"])

    def test_rejects_duplicate_model_features(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_prediction_frame(self.input_values, ["spo2", "spo2"])

    def test_rejects_empty_feature_list(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            build_prediction_frame(self.input_values, [])


class PatientRecordFilterTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "Patient Number": [101, 102, 103],
                "Heart Rate": [70, 80, 90],
            }
        )

    def test_matches_exact_patient_id_and_trims_query(self):
        result = filter_patient_records(self.data, "Patient Number", " 102 ")

        self.assertEqual(result["Heart Rate"].tolist(), [80])

    def test_empty_query_returns_no_records(self):
        result = filter_patient_records(self.data, "Patient Number", " ")

        self.assertTrue(result.empty)

    def test_rejects_missing_patient_column(self):
        with self.assertRaisesRegex(KeyError, "Patient column not found"):
            filter_patient_records(self.data, "Unknown", "101")


class PaginateFrameTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"value": range(7)})

    def test_returns_requested_page(self):
        result = paginate_frame(self.data, page=2, page_size=3)

        self.assertEqual(result["value"].tolist(), [3, 4, 5])

    def test_last_page_may_be_partial(self):
        result = paginate_frame(self.data, page=3, page_size=3)

        self.assertEqual(result["value"].tolist(), [6])

    def test_rejects_invalid_page(self):
        with self.assertRaisesRegex(ValueError, "Page must be greater"):
            paginate_frame(self.data, page=0, page_size=3)

    def test_rejects_invalid_page_size(self):
        with self.assertRaisesRegex(ValueError, "Page size must be greater"):
            paginate_frame(self.data, page=1, page_size=0)


class VerifiedArtifactDownloadTests(unittest.TestCase):
    class HttpsResponse(BytesIO):
        def geturl(self):
            return "https://models.example.test/model.pkl"

    def test_downloads_artifact_when_checksum_matches(self):
        content = b"trusted model bytes"
        checksum = hashlib.sha256(content).hexdigest()

        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "model.pkl"
            with patch(
                "health_utils.urlopen",
                return_value=self.HttpsResponse(content),
            ):
                result = download_verified_artifact(
                    "https://models.example.test/model.pkl",
                    checksum,
                    target,
                )

            self.assertEqual(result.read_bytes(), content)

    def test_rejects_checksum_mismatch_without_creating_target(self):
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "model.pkl"
            with patch(
                "health_utils.urlopen",
                return_value=self.HttpsResponse(b"wrong model bytes"),
            ):
                with self.assertRaisesRegex(ValueError, "checksum does not match"):
                    download_verified_artifact(
                        "https://models.example.test/model.pkl",
                        "0" * 64,
                        target,
                    )

            self.assertFalse(target.exists())

    def test_rejects_non_https_url(self):
        with self.assertRaisesRegex(ValueError, "over HTTPS"):
            download_verified_artifact(
                "http://models.example.test/model.pkl",
                "0" * 64,
                Path("model.pkl"),
            )

    def test_rejects_artifact_larger_than_limit(self):
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "model.pkl"
            with patch(
                "health_utils.urlopen",
                return_value=self.HttpsResponse(b"too large"),
            ):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    download_verified_artifact(
                        "https://models.example.test/model.pkl",
                        hashlib.sha256(b"too large").hexdigest(),
                        target,
                        max_bytes=4,
                    )

            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()