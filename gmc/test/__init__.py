import unittest
import os
import tempfile
import shutil
import importlib
from gmc.conf import settings, ENVIRONMENT_VARIABLE

class TestCase(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.dataset_dir = os.path.realpath(os.path.join(
			tempfile.gettempdir(),
			cls.__name__,
			'dummy_dataset',
		))

		cls.brain_dir = os.path.realpath(os.path.join(
			tempfile.gettempdir(),
			cls.__name__,
			'dummy_brain',
		))

		if not os.path.exists(cls.dataset_dir):
			os.makedirs(cls.dataset_dir)

		if not os.path.exists(cls.brain_dir):
			os.makedirs(cls.brain_dir)

		#create music directories and files
		for gen_dir in settings.GENRES:
			dir_path = os.path.join(cls.dataset_dir, gen_dir)
			if not os.path.exists(dir_path):
				os.makedirs(dir_path)
				for i in range(5):
					with open(os.path.join(dir_path, '%s%d.wav'%(gen_dir, i)), 'w'):
						pass

	@classmethod
	def tearDownClass(cls):
		shutil.rmtree(cls.dataset_dir)
		shutil.rmtree(cls.brain_dir)