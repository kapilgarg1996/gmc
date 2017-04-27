"""
Class MusicSet holds the files and their respective
genres
""" 
from gmc.conf import settings

class MusicSet:

	def __init__(self, force_load=False, genres=None):
		self.dataset_dir = settings.DATASET_DIR
		self.results_dir = settings.BRAIN_DIR
		self.force_load = force_load

	def load_files()