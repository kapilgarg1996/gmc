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
			def __init__(self, force=False):
				self.storage = store('/tmp', force)
			def get_name(self, name='dummy'):
				n = self.storage[cls.dummy_path(0)]
				if n is not None:
					return n
				self.storage[cls.dummy_path(0)] = name
				return name

			def set_name(self, name='dummy'):
				n = self.storage[cls.dummy_path(0)]
				if n is not None:
					self.name = n
					return n
				self.name = name
				self.storage[cls.dummy_path(0)] = name
				return name

			def destory(self):
				os.remove(cls.dummy_path(0))

		cls.Dummy = Dummy

	def test_getter(self):
		dum = self.Dummy()
		name = dum.get_name('custom_name')
		self.assertEqual(name, 'custom_name')
		name = dum.get_name('another_name')

		#this shows that old saved copy is used
		self.assertEqual(name, 'custom_name')
		dum.destory()

	def test_setter(self):
		dum = self.Dummy()
		dum.set_name('custom_name')
		self.assertEqual(dum.name, 'custom_name')
		dum.set_name('different_name')
		self.assertEqual(dum.name, 'custom_name')
		dum.destory()

	def test_force_reloading(self):
		dum = self.Dummy(force=True)
		name = dum.get_name('custom_name')
		self.assertEqual(name, 'custom_name')
		name = dum.get_name('another_name')

		#this shows that new copy is made
		self.assertEqual(name, 'another_name')
		dum.destory()

	@classmethod
	def dummy_path(cls, name):
		cls._dummy_paths[name] = os.path.join('/tmp', 'dummy'+str(name)+'.dat')
		return cls._dummy_paths[name]

	@classmethod
	def tearDownClass(cls):
		for path in cls._dummy_paths:
			if os.path.isfile(cls._dummy_paths[path]):
				os.remove(cls._dummy_paths[path])