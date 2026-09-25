import sys
print(f"Python: {sys.version}")
try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError:
    print("numpy: NOT INSTALLED")
try:
    import gzip
    print("gzip: ok")
except ImportError:
    print("gzip: NOT INSTALLED")
