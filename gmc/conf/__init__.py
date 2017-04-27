import os
import importlib
from gmc.conf import global_settings

ENVIRONMENT_VARIABLE = "GMC_SETTINGS_MODULE"
class Settings:
	"""
	Module to load settings to configure gmc
	"""
	def __init__(self, *args, **kwargs):
		self.settings = None
		self.settings_module = None

	def __getattr__(self, name):
		"""
		Make settings  available as the attributes.
		Like settings.DATASET_DIR
		"""
		settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
		if self.settings is None or settings_module != self.settings_module:
			self.load_settings()
		return self.settings[name]

	def load_settings(self):
		self.settings = {}
		for setting in dir(global_settings):
			if setting.isupper():
				self.settings[setting] = getattr(global_settings, setting)
		self.settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
		mod = importlib.import_module(self.settings_module)

		for setting in dir(mod):
			if setting.isupper():
				self.settings[setting] = getattr(mod, setting)

settings = Settings()