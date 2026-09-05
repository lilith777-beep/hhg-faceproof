import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_evidence_schema_is_valid() -> None:
    schema_path = Path(__file__).parents[1] / "schema" / "evidence-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
