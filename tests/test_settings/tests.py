import unittest
import os
from gmc.conf import settings

class TestSettings(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("GMC_SETTINGS_MODULE", "test_settings.setting")

    def test_settings_loader(self):
        self.assertEqual(settings.DATASET_DIR, '/home/kapil')