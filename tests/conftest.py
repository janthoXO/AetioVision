import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # `from fakes import ...`

MOCK = Path(__file__).with_name("mock_terminology.json")
if not MOCK.exists():  # SNOMED/LOINC excerpt is licensed, not in git (README "Tests")
    collect_ignore = ["test_catalog.py", "test_contracts.py", "test_extraction.py", "test_pipeline.py", "test_resolution.py"]

    def pytest_terminal_summary(terminalreporter):
        terminalreporter.write_line(f"SKIPPED {len(collect_ignore)} terminology test files: {MOCK.name} missing, "
                                    "run `uv run python tests/fetch_mock_terminology.py`", yellow=True)
