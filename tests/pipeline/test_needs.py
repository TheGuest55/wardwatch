"""tests/needs.py: skip reasons name the missing data and the manual level; present data never skips."""
import unittest

from tests import needs as n


class TestNeeds(unittest.TestCase):
    def test_every_set_has_paths_a_description_and_a_manual_reference(self):
        for name, (paths, what, where) in n.SETS.items():
            self.assertTrue(paths and what and where.startswith("USER_MANUAL"), name)

    def test_reason_names_what_is_missing_and_the_manual_section(self):
        orig = n.SETS["CIC_CSV"]
        try:
            n.SETS["CIC_CSV"] = (["definitely/not/here"], "CICIoMT2024's feature CSVs (not redistributed)", "USER_MANUAL section 5")
            self.assertEqual(n.missing("CIC_CSV"), ["CIC_CSV"])
            r = n.reason("CIC_CSV")
            self.assertIn("CICIoMT2024's feature CSVs", r)
            self.assertIn("USER_MANUAL section 5", r)
        finally:
            n.SETS["CIC_CSV"] = orig

    def test_present_data_means_no_skip_and_empty_reason(self):
        orig = n.SETS["DATASET"]
        try:
            n.SETS["DATASET"] = (["tests/needs.py"], "x", "USER_MANUAL Level 2")
            self.assertEqual(n.missing("DATASET"), [])
            self.assertEqual(n.reason("DATASET"), "")
        finally:
            n.SETS["DATASET"] = orig

    def test_every_manual_level_cited_in_a_skip_message_exists_in_the_manual(self):
        text = (n.ROOT / "USER_MANUAL.md").read_text()
        for name, (_, _, where) in n.SETS.items():
            heading = where.replace("USER_MANUAL ", "")
            self.assertIn(heading.split()[0] + " " + heading.split()[1] if heading.startswith("Level") else "5. Experiments", text, name)


if __name__ == "__main__":
    unittest.main()
