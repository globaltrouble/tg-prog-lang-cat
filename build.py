#!/usr/bin/env python3
"""
Build script. Performs shared library build, links it with tester binary and finaly builds tester binary.
"""

import argparse
import os
import shutil
from subprocess import check_call


DEFAULT_SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD_PATH = os.path.join(DEFAULT_SOURCE_PATH, "build")

LIB_BINARY_NAME = "libtglang.so"

LIB_TARGET = "libtglang"
TESTER_TARGET = "tglang-tester"
LINK_TARGET = "link-tglang"
AVAILABLE_TARGETS = (
    LIB_TARGET,
    TESTER_TARGET,
)


def main():
    args = parse_args()
    if args.clean and os.path.exists(args.build_dir):
        shutil.rmtree(args.build_dir)
    
    context = {
        "build_dir": args.build_dir,
        "source_dir": args.source_dir,
    }
    
    visited = set()
    def traverse(target):
        if target in visited:
            return
        visited.add(target)
        
        for dep in DEPENDENCIES[target]:
            traverse(dep)
        
        ACTIONS[target](target, context)
    
    for t in args.target:
        traverse(t)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        help="Targets to build",
        choices=[LIB_TARGET, TESTER_TARGET],
        default=[TESTER_TARGET],
        nargs="*",
    )
    parser.add_argument(
        "--clean",
        help="Remove cmake config dir to rebuild project from scratch",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-S", 
        "--source-dir", 
        default=DEFAULT_SOURCE_PATH, 
        help="Project source directory"
    )
    parser.add_argument(
        "-B",
        "--build-dir",
        default=DEFAULT_BUILD_PATH,
        help="Cmake build directory, stores CMake generated output and build artficats",
    )
    return parser.parse_args()


def build_target(target, context):
    build_dir = os.path.join(context["build_dir"], target)
    source_dir = os.path.join(context["source_dir"], target)
    for d in build_dir, source_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    check_call(["cmake", "-S", source_dir, "-B", build_dir])
    check_call(["cmake", "--build", build_dir, "--parallel", "-v"])


def link_lib(_target, context):
    src = os.path.join(context["build_dir"], LIB_TARGET, LIB_BINARY_NAME)
    dst = os.path.join(context["source_dir"], TESTER_TARGET, LIB_BINARY_NAME)
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src=src, dst=dst, target_is_directory=True)
    

DEPENDENCIES = {
    TESTER_TARGET: [LINK_TARGET],
    LINK_TARGET: [LIB_TARGET],
    LIB_TARGET: [],
}


ACTIONS = {
    TESTER_TARGET: build_target,
    LINK_TARGET: link_lib,
    LIB_TARGET: build_target,
}


if __name__ == "__main__":
    main()
