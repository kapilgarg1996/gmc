import unittest
import os
import tempfile
import shutil
import importlib
from gmc.conf import settings, ENVIRONMENT_VARIABLE

class TestCase(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		super(TestCase, cls).setUpClass()
		cls._old_settings = {}

		for setting, value in settings:
			cls._old_settings[setting] = value

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

		cls.settings_file = os.path.realpath(os.path.join(
			tempfile.gettempdir(),
			cls.__name__,
			'settings.py',
		))

		with open(cls.settings_file, 'w') as set_file:
			set_file.write("DATASET_DIR = '%s'\n" % cls.dataset_dir)
			set_file.write("BRAIN_DIR = '%s'" % cls.brain_dir)

		settings.modify({
			'DATASET_DIR' : cls.dataset_dir,
			'BRAIN_DIR' : cls.brain_dir
		})

	@classmethod
	def tearDownClass(cls):
		os.remove(cls.settings_file)
		shutil.rmtree(cls.dataset_dir)
		shutil.rmtree(cls.brain_dir)

		settings.modify(cls._old_settings)
		super(TestCase, cls).tearDownClass()