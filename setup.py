from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent

VERSION = '0.0.6'
DESCRIPTION = 'Pytrino: A new way to numerically compute neutrino oscillations'

LONG_DESCRIPTION = (this_directory / "README.md").read_text()

# Setting up
setup(
        name="pytrino", 
        version=VERSION,
        author="Baalateja Kataru",
        author_email="<kavesbteja@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=['numpy'], 

        keywords=['probability', 'neutrino', 'oscillations'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)