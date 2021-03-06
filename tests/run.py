import argparse
import unittest
import os
import importlib
import sys

from gmc.conf import settings, ENVIRONMENT_VARIABLE
from gmc.core import handler

def build_suite(test_labels=None):
    suite = unittest.TestSuite()
    test_loader = unittest.defaultTestLoader
    test_labels = test_labels or ['.']
    discover_kwargs = {}
    for label in test_labels:
        kwargs = discover_kwargs.copy()
        tests = None

        label_as_path = os.path.abspath(label)
        # if a module, or "module.ClassName[.method_name]", just run those
        if not os.path.exists(label_as_path):
            tests = test_loader.loadTestsFromName(label)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modules', nargs='*',
        help='Optional path(s) to test modules; e.g. "test_settings" or '
             '"test_settings.tests.TestSettings.test_settings_loader".',
    )
    parser.add_argument('--settings', help='Test gmc with different settings file')
    args = parser.parse_args()

    if args.settings:
        handler.execute_from_command_line(['', args.settings], quiet=True)
        os.environ['DUMMY'] = "FALSE"
    else:
        os.environ[ENVIRONMENT_VARIABLE] = 'setting'
        os.environ['DUMMY'] = "TRUE"

    args.modules = [os.path.normpath(labels) for labels in args.modules]
    suite = build_suite(args.modules)
    runner = unittest.TextTestRunner()
    runner.run(suite)