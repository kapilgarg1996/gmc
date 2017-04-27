"""
Command handler module for GMC
"""

import sys
import importlib
import os
from gmc.conf import ENVIRONMENT_VARIABLE
from gmc.conf import settings

def execute_from_command_line(argv=None, quiet=False):
	"""
	try to load a module specified in the given path.
	Set GMC_SETTINGS_MODULE environment variable
	"""
	argv = argv or sys.argv[:]
	if(len(argv) != 2) and not quiet:
		print('Incorrect Usage of gmc-main command')
		return

	if os.path.exists(argv[1]):
		module_file = os.path.basename(argv[1])
		module_path = os.path.abspath(argv[1])
		module_dir = os.path.dirname(module_path)
		sys.path.append(module_dir)
		module_name = module_file.split('.py')[0]
		os.environ[ENVIRONMENT_VARIABLE] = module_name
	else:
		if not quiet:
			print('Incorrect format for settings.py path')
	
	if not quiet:
		try:
			print("Dataset Directory set to '%s'" % settings.DATASET_DIR)
			print("Results Directory set to '%s'" % settings.BRAIN_DIR)
		except AttributeError:
			print('Could not load settings')