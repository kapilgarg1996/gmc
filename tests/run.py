import unittest
import os
import importlib

def build_suite():
	suite = unittest.TestSuite()
	test_loader = unittest.defaultTestLoader
	test_labels = ['.']
	discover_kwargs = {}
	for label in test_labels:
		kwargs = discover_kwargs.copy()
		tests = None

		label_as_path = os.path.abspath(label)
		# if a module, or "module.ClassName[.method_name]", just run those
		if not os.path.exists(label_as_path):
			tests = self.test_loader.loadTestsFromName(label)
		elif os.path.isdir(label_as_path):
			top_level = label_as_path
			while True:
				init_py = os.path.join(top_level, '__init__.py')
				if os.path.exists(init_py):
					try_next = os.path.dirname(top_level)
					if try_next == top_level:
						# __init__.py all the way down? give up.
						break
					top_level = try_next
					continue
				break
			kwargs['top_level_dir'] = top_level

		if not (tests and tests.countTestCases()) and is_discoverable(label):
			# Try discovery if path is a package or directory
			tests = test_loader.discover(start_dir=label, **kwargs)

			# Make unittest forget the top-level dir it calculated from this
			# run, to support running tests from two different top-levels.
			test_loader._top_level_dir = None

		suite.addTests(tests)

	return suite

def is_discoverable(label):
	"""
	Check if a test label points to a python package or file directory.
	Relative labels like "." and ".." are seen as directories.
	"""
	try:
		mod = importlib.import_module(label)
	except (ImportError, TypeError):
		pass
	else:
		return hasattr(mod, '__path__')

	return os.path.isdir(os.path.abspath(label))

if __name__ == '__main__':
	suite = build_suite()
	runner = unittest.TextTestRunner()
	runner.run(suite)