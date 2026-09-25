import json
import tempfile
import unittest
from pathlib import Path

from research import socialmembench_episode_official_harness as run


class FrozenVerificationTest(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"abc")
            self.assertEqual(
                run._sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class ReportTest(unittest.TestCase):
    def test_all_four_conditions_are_declared(self):
        self.assertEqual(
            run.ALL_CONDITIONS,
            ("RAW", "RAW+FLAT", "RAW+VERSIONED", "RAW+EPISODES"),
        )


if __name__ == "__main__":
    unittest.main()
