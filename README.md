# gmc
Genre Based Music Classifier

## Installation

Follow the below commands to install gmc on your system
```bash
$ `git clone https://github.com/kapilgarg1996/gmc.git`
$ `sudo pip install gmc`
```
## Environment Setup for Development

* install `pip` (Google how to install pip on your system)
* install `virtualenv` (Google how to install virtualenv)
* Now follow the below commands
  ``` bash
  $ `virtualenv env`
  $ `cd env`
  $ `source bin/activate`
  $ `git clone https://github.com/kapilgarg1996/gmc.git`
  $ `cd gmc`
  $ `pip install -e .`
  ```
## Run Tests

GMC has its own test running module. Follow the below test
to run the test suite.

```bash
$ `cd tests`
$ `python run.py`
```

## Writing Tests

Its a good practice to write tests before adding a new feature.

#### Adding Tests for a new module

* Create a module inside the `tests` module with name starting 
  with test like `test_module`
* Create `__init__.py` inside your `test_module`
* Add tests file following the [unittest](https://docs.python.org/3/library/unittest.html) convention

#### Adding Tests for a new feature

* Go to the test module which tests the module to which the new feature
  is added. Eg. if you add a new feature to `Settings` then go to
  `test_settings` module inside `tests`

* Write a new method or a new test class or a new test file depending
  on the size of the feature.

## Documentation

GMC uses sphinx for its documentation. The documentation is present
inside `docs/` folder and is in plain text format. Go to the `docs/`
folder where You can build the documentation using the following
command

```bash
make html
```

Here `html` is the target format. The built documentation will be
present in `docs/build/html`. For a complete list of available
formats, visit [sphinx build options](http://www.sphinx-doc.org/en/stable/invocation.html#invocation-of-sphinx-build)