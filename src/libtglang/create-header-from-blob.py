#!/usr/bin/env python3
import argparse


def main():
    args = parse_args()
    generate_cpp_header(
        file_path=args.file,
        header=args.header,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        help="Target file path to encode to header.",
        required=True,
    )
    parser.add_argument(
        "--header",
        help="Header name (without .h suffix) and const char variable name to create.",
        required=True,
    )
    return parser.parse_args()


def generate_cpp_header(file_path, header):
    with open(file_path, 'rb') as binary_file:
        hex_string = ',\n\t'.join("'\\x{:02x}'".format(bt) for bt in binary_file.read())

    header_content = f"""
#ifndef {header}_H
#define {header}_H

const char {header}[] = {{
\t{hex_string}
}};

#endif // {header}_H"""

    header_file_path = f"{header}.h"
    with open(header_file_path, 'w') as header_file:
        header_file.write(header_content)


if __name__ == "__main__":
    main()
