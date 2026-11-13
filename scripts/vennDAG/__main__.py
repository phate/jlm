#!/usr/bin/env python3
"""
Entry point for running vennDAG as a module or directory.
e.g. python3 -m vennDAG input.dot or python3 scripts/vennDAG input.dot
"""

import sys
import os

# Ensure package is on sys.path
pkg_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(pkg_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from vennDAG.main import main

if __name__ == "__main__":
    sys.exit(main())
