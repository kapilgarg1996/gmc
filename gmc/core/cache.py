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
    def __init__(self, path, force=False):
        self.path = path
        self.data = {}
        self.force = force

    def __getitem__(self, name):
        filepath = os.path.join(self.path, name)
        if self.force:
            return None
        if name not in self.data:
            try:
                with open(filepath, 'rb') as f:
                    self.data[name] = pickle.load(f)
                return self.data[name]
            except Exception:
                return None
        return self.data[name]

    def __setitem__(self, name, value):
        filepath = os.path.join(self.path, name)
        self.data[name] = value
        if self.force or not os.path.isfile(filepath):
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)