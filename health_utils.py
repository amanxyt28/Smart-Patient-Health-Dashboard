from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import urlopen

import pandas as pd


def build_prediction_frame(
    input_values: Mapping[str, float],
    features: Sequence[str],
) -> pd.DataFrame:
    ordered_features = list(features)
    if not ordered_features:
        raise ValueError("At least one model feature is required.")
    if len(set(ordered_features)) != len(ordered_features):
        raise ValueError("Model features must be unique.")

    missing_features = [
        feature for feature in ordered_features if feature not in input_values
    ]
    if missing_features:
        raise ValueError(
            "Missing input values for model features: "
            + ", ".join(missing_features)
        )

    return pd.DataFrame(
        [{feature: input_values[feature] for feature in ordered_features}],
        columns=ordered_features,
    )


def filter_patient_records(
    data: pd.DataFrame,
    patient_column: str,
    patient_id: str,
) -> pd.DataFrame:
    if patient_column not in data.columns:
        raise KeyError(f"Patient column not found: {patient_column}")

    normalized_id = str(patient_id).strip()
    if not normalized_id:
        return data.iloc[0:0]

    patient_ids = data[patient_column].astype("string").str.strip()
    return data.loc[patient_ids.eq(normalized_id)]


def paginate_frame(
    data: pd.DataFrame,
    page: int,
    page_size: int,
) -> pd.DataFrame:
    if page < 1:
        raise ValueError("Page must be greater than or equal to 1.")
    if page_size < 1:
        raise ValueError("Page size must be greater than or equal to 1.")

    start = (page - 1) * page_size
    return data.iloc[start:start + page_size]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_artifact(
    url: str,
    expected_sha256: str,
    target_path: Path,
    max_bytes: int = 1024 * 1024 * 1024,
) -> Path:
    if urlsplit(url).scheme != "https":
        raise ValueError("Model artifacts must be downloaded over HTTPS.")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF"
        for character in expected_sha256
    ):
        raise ValueError("A valid SHA-256 checksum is required.")
    if max_bytes <= 0:
        raise ValueError("Maximum artifact size must be positive.")

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total_bytes = 0
    temporary_path = None

    try:
        with NamedTemporaryFile(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(url, timeout=60) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Model artifact redirects must use HTTPS.")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ValueError("Model artifact exceeds the size limit.")
                    digest.update(chunk)
                    temporary_file.write(chunk)

        if digest.hexdigest().lower() != expected_sha256.lower():
            raise ValueError("Model artifact SHA-256 checksum does not match.")

        temporary_path.replace(target_path)
        return target_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise