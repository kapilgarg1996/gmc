"""
Module to test gmc-main command.
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess
from gmc.conf import settings, ENVIRONMENT_VARIABLE
from gmc.test import TestCase

class TestGMCMain(TestCase):
	def test_gmc_main_settings(self):
		old_cwd = os.getcwd()
		os.chdir(os.path.dirname(self.settings_file))
		out, err = subprocess.Popen(
			['gmc-main', self.settings_file],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			universal_newlines=True,
		).communicate()
		os.chdir(old_cwd)
		self.assertTrue(self.dataset_dir in out)
		self.assertTrue(self.brain_dir in out)