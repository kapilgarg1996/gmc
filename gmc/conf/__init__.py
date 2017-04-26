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

	def __getattr__(self, name):
		"""
		Make settings  available as the attributes.
		Like settings.DATASET_DIR
		"""
		if self.settings is None:
			self.load_settings()
		return self.settings[name]

	def load_settings(self):
		self.settings = {}
		for setting in dir(global_settings):
			if setting.isupper():
				self.settings[setting] = getattr(global_settings, setting)
		settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
		mod = importlib.import_module(settings_module)

		for setting in dir(mod):
			if setting.isupper():
				self.settings[setting] = getattr(mod, setting)

settings = Settings()