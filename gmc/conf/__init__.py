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
        self.load_settings()
        return self.settings[name]

    def __iter__(self):
        self.load_settings()
        return iter(self.settings)

    def load_settings(self):
        settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
        if self.settings is not None and settings_module == self.settings_module:
            return

        self.settings = {}
        for setting in dir(global_settings):
            if setting.isupper():
                self.settings[setting] = getattr(global_settings, setting)
        self.settings_module = os.environ.get(ENVIRONMENT_VARIABLE, None)
        
        if self.settings_module is not None:
            mod = importlib.import_module(self.settings_module)

            for setting in dir(mod):
                if setting.isupper():
                    self.settings[setting] = getattr(mod, setting)

    def modify(self, new_settings):
        for name in new_settings:
            if name in self.settings:
                self.settings[name] = new_settings[name]

settings = Settings()