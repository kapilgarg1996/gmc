import unittest
import os
from gmc.conf import settings
from gmc.core import handler

class TestSettings(unittest.TestCase):
	def setUp(self):
		handler.execute_from_command_line(['', 'setting.py'], quiet=True)

	def test_settings_loader(self):
		self.assertEqual(settings.DATASET_DIR, '/home/kapil')