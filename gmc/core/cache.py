import pickle
import os

class store:
	"""
	This is a decorator class to store data from a function. It bypass
	function execution if data file is found. Force reloading is provided
	when class attributes changes and new data needs to be loaded irrespective
	of already existing data store
	Usage:
	if 'prop' is used then object's attribute 'prop' is set to result which is either
	one of the following
	1. method return value
	2. object's prop value computed inside method

	if 'prop' not used then function's return value is stored in file
	and returned.
	"""
	def __init__(self, path, prop=None, force=False):
		self.path = path
		self.prop = prop
		self.force = force

	def __call__(self, func):
		self.func = func
		def decorator(*args, **kwargs):
			result = None
			store_found = False
			return_val = None
			if os.path.isfile(self.path) and not self.force:
				with open(self.path, 'rb') as f:
					result = pickle.load(f)
				store_found = True
			else:
				result = return_val = self.func(*args, **kwargs)
				if result is None:
					obj = args[0]
					result = getattr(obj, self.prop)

			if not store_found:
				with open(self.path, 'wb') as f:
					pickle.dump(result, f)

			if self.prop:
				obj = args[0]
				setattr(obj, self.prop, result)
			
			if not store_found:
				return return_val

			if not self.prop:
				return result
		return decorator