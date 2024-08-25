from setuptools import setup, find_packages, Extension
from pathlib import Path
try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

VERSION = '0.0.1'
DESCRIPTION = 'Pytrino: A new way to numerically compute neutrino oscillations'
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text()

ext = '.pyx' if USE_CYTHON else '.c'
names = ["two_flavor_matter", "three_flavor_matter"]
package  = "pytrino"
dist = "src"

extnames = [".".join([package, name]) for name in names]
extpaths = [str(path) for path in [Path(dist).joinpath(Path(package)).joinpath(Path(name + ext)) for name in names]]
extensions = [Extension(name, [path], optional=False) for name, path in zip(extnames, extpaths)]

if USE_CYTHON:
    extensions = cythonize(extensions)

# Setting up
setup(
        name="pytrino", 
        version=VERSION,
        author="BKModding",
        author_email="<kavesbteja@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=find_packages(where="src"),
        package_dir={'': 'src'},
        python_requires='>=3.5',
        platforms=['Windows', 'Linux', 'Mac OS X'],
        url='https://github.com/BK-Modding/pytrino',
        download_url='https://github.com/BK-Modding/pytrino',
        license='MIT',
        install_requires=[],
        extras_require={
            'dev': [
                'numpy',
                'matplotlib'
                'Cython'
            ],
            'demo': 'numpy'
        },
        ext_modules = extensions,
        keywords = [
            "neutrino oscillation",
            "matter effects",
            "constant density",
            "linear algebra",
            "Cython",
            "scientific computing",
            "physics",
            "quantum mechanics",
            "numerical simulations",
            "mathematical modeling",
            "computational physics",
            "particle physics",
            "MSW effect",
            "probability",
            "eigenvalue",
            "eigenvector",
            "identity",
        ],
        classifiers = [
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Framework :: Matplotlib",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English"
        ]
    )

