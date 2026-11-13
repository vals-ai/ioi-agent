import tempfile
import unittest
from pathlib import Path

from watch_stage9 import optional_fits, read


class SchedulerTests(unittest.TestCase):
    def test_optional_deadline_preserves_dev_budget(self):
        fits, seconds = optional_fits(0, 10000, 1000000)
        self.assertTrue(fits)
        self.assertGreaterEqual(seconds, 600)
        self.assertFalse(optional_fits(9000, 10000, 1000000)[0])
        self.assertFalse(optional_fits(0, 10000, 1000000000)[0])

    def test_missing_empty_and_partial_state(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "state.json"
            self.assertIsNone(read(path))
            path.write_text("")
            self.assertIsNone(read(path))
            path.write_text('{"partial":')
            self.assertIsNone(read(path))
            path.write_text('{"done":true}')
            self.assertEqual(read(path), {"done": True})


if __name__ == "__main__":
    unittest.main()
