from pathlib import Path

def pathmaker(filenames, ext):
    paths = [Path(name + ext) for name in filenames]
    paths = [Path("pytrino").joinpath(path) for path in paths]
    paths = [str(path) for path in paths]
    
    return paths

ext = '.pyx'
filenames = ["two_flavor_matter", "three_flavor_matter"]
filepaths = pathmaker(filenames, ext)

print(f"Attempting to compile {', '.join(filepaths)} to .c files...")

try:
    from Cython.Build import cythonize
    cythonize(filepaths)

    compiledpaths = pathmaker(filenames, '.c')
    print(f"Successfully compiled to {', '.join(compiledpaths)}")

except ImportError:
    raise Exception("Cython is not installed or a C compiler is missing. Please install Cython or install a C compiler and try again.")