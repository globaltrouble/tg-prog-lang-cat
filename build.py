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
TESTER_BINARY_NAME = "tglang-tester"

LIB_TARGET = "libtglang"
TESTER_TARGET = "tglang-tester"
LINK_TARGET = "link-tglang"
TESTFILE_TARGET = "test_file"
AVAILABLE_TARGETS = (
    LIB_TARGET,
    TESTER_TARGET,
    TESTFILE_TARGET,
)


def main():
    args = parse_args()
    if args.clean and os.path.exists(args.build_dir):
        shutil.rmtree(args.build_dir)
    
    context = {
        "build_dir": args.build_dir,
        "source_dir": args.source_dir,
        "test_file": args.test_file,
    }
    
    visited = set()
    def traverse(target):
        if target in visited:
            return
        visited.add(target)
        
        for dep in DEPENDENCIES[target]:
            traverse(dep)
        
        print(f"Begin to exec target: `{target}`")
        ACTIONS[target](target, context)
    
    targets = set(args.target)

    if args.test_file:
        targets.add(TESTFILE_TARGET)
        binary_exists = os.path.exists(os.path.join(context["build_dir"], TESTER_TARGET, TESTER_BINARY_NAME))
        if TESTER_TARGET not in targets and not binary_exists:
            targets.add(TESTFILE_TARGET)
    
    for t in targets:
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
    parser.add_argument(
        "-t",
        "--test-file",
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
    check_call(["cmake", "--build", build_dir, "--parallel"])


def link_lib(_target, context):
    src = os.path.join(context["build_dir"], LIB_TARGET, LIB_BINARY_NAME)
    dst = os.path.join(context["source_dir"], TESTER_TARGET, LIB_BINARY_NAME)
    try:
        # can't detect wiether dst exists without exception, so remove it in anycase and ignore exception
        os.remove(dst)
    except FileNotFoundError:
        pass

    os.symlink(src=src, dst=dst)


def run_tester(_target, context):
    binary = src = os.path.join(context["build_dir"], TESTER_TARGET, TESTER_BINARY_NAME)
    if not os.path.exists(binary):
        raise RuntimeError(f"Binary doesn't exists: `{binary}`")
    check_call([binary, context["test_file"]])


DEPENDENCIES = {
    TESTER_TARGET: [LINK_TARGET],
    LINK_TARGET: [LIB_TARGET],
    LIB_TARGET: [],
    TESTFILE_TARGET: [TESTER_TARGET],
}


ACTIONS = {
    TESTER_TARGET: build_target,
    LINK_TARGET: link_lib,
    LIB_TARGET: build_target,
    TESTFILE_TARGET: run_tester,
}


if __name__ == "__main__":
    main()
