from setuptools import find_packages, setup

setup(
    name='gmc',
    version='1.0.0.dev1',
    url='https://github.com/kapilgarg1996/gmc',
    author='Kapil GC',
    author_email='kapilgarg1996@gmail.com',
    description=('A genre based music classifier project '
                 'using neural networks'),
    license='MIT',
    packages=find_packages(exclude=['docs', 'bin', 'tests*']),
    scripts=['gmc/bin/gmc-main.py'],
    entry_points={'console_scripts': [
        'gmc-main = gmc.core.handler:execute_from_command_line',
    ]},
    install_requires=['tensorflow', 'librosa'],
    extras_require={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
)