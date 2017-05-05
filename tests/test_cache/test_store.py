import unittest
import os
import tempfile
from gmc.core.cache import store

class TestStore(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._dummy_paths = {}
		cls.next_path = 0
		class Dummy:
			@store(cls.dummy_path(0), 'name')
			def get_name(self, name=None):
				return name or 'dummy'

			@store(cls.dummy_path(1), 'name')
			def set_name(self, name=None):
				self.name = name or 'dummy'

		cls.Dummy = Dummy

	def test_with_prop_with_return(self):
		dum = self.Dummy()
		name = dum.get_name('custom_name')
		self.assertEqual(dum.name, 'custom_name')
		self.assertEqual(name, 'custom_name')

		name = dum.get_name('different_name')
		# The following assertions show that 'name' is loaded from stored file
		# and function call is bypassed 
		self.assertEqual(dum.name, 'custom_name')
		self.assertNotEqual(name, 'custom_name')

	def test_with_prop_without_return(self):
		dum = self.Dummy()
		dum.set_name()
		self.assertEqual(dum.name, 'dummy')
		dum.set_name('different_name')
		self.assertEqual(dum.name, 'dummy')

	def test_without_prop_with_return(self):
		@store(self.dummy_path(2))
		def dummy_func(name=None):
			return name or 'dummy'

		result = dummy_func()
		self.assertEqual(result, 'dummy')

		next_result = dummy_func('different_name')
		self.assertEqual(result, 'dummy')

	def test_force_reloading(self):
		@store(self.dummy_path(2), force=True)
		def dummy_func(name=None):
			return name or 'dummy'

		result = dummy_func('not_dummy')
		self.assertEqual(result, 'not_dummy')

		#remove dummy path file for other tests which don't use
		#force reloading
		os.remove(self.dummy_path(2))

	@classmethod
	def dummy_path(cls, name):
		cls._dummy_paths[name] = os.path.join('/tmp', 'dummy'+str(name)+'.dat')
		return cls._dummy_paths[name]

	@classmethod
	def tearDownClass(cls):
		for path in cls._dummy_paths:
			if os.path.isfile(cls._dummy_paths[path]):
				os.remove(cls._dummy_paths[path])