import unittest

from watch_stage8 import next_limit


class ControlTests(unittest.TestCase):
    def test_load_memory_rate_and_floor(self):
        self.assertEqual(next_limit(16, 16, None, 1, 10, 4)[0], 16)
        self.assertEqual(next_limit(16, 16, None, 1, 10, 15)[0], 8)
        self.assertEqual(next_limit(12, 12, None, 1, 10, 4, memory=True)[0], 6)
        self.assertEqual(next_limit(4, 4, None, 1, 10, 4, rate=True)[0], 2)
        self.assertEqual(
            next_limit(1, 4, "stage8_rate_load_memory_backoff", 1, 10, 15)[0], 1
        )
        self.assertEqual(next_limit(0, 16, "stage8_15_45_utc_cutoff", 1, 10, 15)[0], 0)

    def test_cutoff_overrides_every_control(self):
        self.assertEqual(
            next_limit(16, 16, None, 10, 10, 1), (0, "stage8_15_45_utc_cutoff")
        )
        self.assertEqual(next_limit(3, 16, "user_override", 11, 10, 1)[0], 0)

    def test_does_not_override_user_resize(self):
        self.assertEqual(next_limit(3, 16, None, 1, 10, 15)[0], 3)
        self.assertEqual(next_limit(4, 4, "user_override", 1, 10, 15, rate=True)[0], 4)


if __name__ == "__main__":
    unittest.main()
