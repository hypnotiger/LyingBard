import atexit
import sys

atexit.register(print, "exit")
sys.exit(3)