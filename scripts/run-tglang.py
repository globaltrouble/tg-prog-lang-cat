#!/usr/bin/env python3

import sys
import os
from subprocess import check_call

SOURCE_FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def show_help():
    print(
    """Usage: `progname runner ...args-to-runner`
        Examples:
            - `./progname single /path/to/file`
            - `./progname multi /path/to/file1 /path/to/file2 ... /path/to/fileN`
    """
    )

if len(sys.argv) < 3:
    show_help()
    raise RuntimeError("Not enough arguments")

cmd, *args = sys.argv[1:]
if cmd == "single":
    assert len(args) == 1, "Wrong number of args for tglang-tester single runner"
    binary = "tglang-tester"
elif cmd == "multi":
    assert len(args) > 0, "Wrong number of args for tglang-tester multi runner"
    binary = "tglang-multitester"
else:
    show_help()
    raise RuntimeError(f"Unknown cmd `{cmd}`")

check_call([os.path.join(SOURCE_FILE_DIR_PATH, binary)] + args, env={"LD_LIBRARY_PATH": SOURCE_FILE_DIR_PATH})

