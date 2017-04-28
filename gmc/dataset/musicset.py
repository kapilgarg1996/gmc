"""
Class MusicSet holds the files and their respective
genres
""" 
import os
from gmc.conf import settings

class MusicSet:
	def __init__(self, force_load=False, genres=None):
		self.dataset_dir = settings.DATASET_DIR
		self.results_dir = settings.BRAIN_DIR
		self.force_load = force_load
		self.genres = genres or settings.GENRES
		self.files = {}

	def load_files(self):
		for genre in self.genres:
			genre_path = os.path.join(self.dataset_dir, genre)
			if os.path.isdir(genre_path):
				self.files[genre] = []
				for f in os.listdir(genre_path):
					file_path = os.path.join(genre_path, f)
					if os.path.isfile(file_path) and f.endswith(".wav"):
						self.files[genre].append(file_path)