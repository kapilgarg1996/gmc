import unittest
import os
from gmc.conf import settings

class TestSettings(unittest.TestCase):
	
    def test_settings_loader(self):
        self.assertEqual(settings.DATASET_DIR, '/home/kapil')