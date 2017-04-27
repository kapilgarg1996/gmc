"""
Module to test gmc-main command.
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess
from gmc.conf import settings
from gmc.test import TestCase

class TestGMCMain(TestCase):

	@classmethod
	def setUpClass(cls):
		super(TestGMCMain, cls).setUpClass()
		cls.settings_file = os.path.realpath(os.path.join(
			tempfile.gettempdir(),
			cls.__name__,
			'settings.py',
		))

		with open(cls.settings_file, 'w') as set_file:
			set_file.write("DATASET_DIR = '%s'\n" % cls.dataset_dir)
			set_file.write("BRAIN_DIR = '%s'" % cls.brain_dir)

	@classmethod
	def tearDownClass(cls):
		os.remove(cls.settings_file)
		super(TestGMCMain ,cls).tearDownClass()

	def test_gmc_main_settings(self):
		out, err = subprocess.Popen(
			['gmc-main', self.settings_file],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			universal_newlines=True,
		).communicate()

		self.assertTrue(self.dataset_dir in out)
		self.assertTrue(self.brain_dir in out)