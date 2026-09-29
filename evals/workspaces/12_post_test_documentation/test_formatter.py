import unittest

from formatter import normalize_title


class NormalizeTitleTests(unittest.TestCase):
    def test_collapses_repeated_spaces(self):
        self.assertEqual(normalize_title("  Project   status  "), "Project status")

    def test_collapses_tabs_and_newlines(self):
        self.assertEqual(normalize_title("Weekly\treport\nready"), "Weekly report ready")

    def test_blank_input(self):
        self.assertEqual(normalize_title(" \t\n "), "")

    def test_preserves_case_and_punctuation(self):
        self.assertEqual(normalize_title("Release: v2.1!"), "Release: v2.1!")


if __name__ == "__main__":
    unittest.main()
