import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardAppTests(unittest.TestCase):
    def test_all_sections_render_without_streamlit_exceptions(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=20)
        sections = [
            "📊 Dashboard Overview",
            "🔎 Patient Explorer",
            "❤️ Vital Analytics",
            "🩺 Disease Analytics",
            "⚠️ Health Alerts",
            "🤖 ML Prediction",
            "📋 Dataset Explorer",
        ]

        self.assertEqual(app.title[0].value, "Patient Health Record Dashboard")
        for section in sections:
            with self.subTest(section=section):
                app.radio[0].set_value(section).run(timeout=20)
                self.assertFalse(app.exception)
                if section == "📋 Dataset Explorer":
                    self.assertLessEqual(len(app.dataframe[0].value), 100)


if __name__ == "__main__":
    unittest.main()