#!/usr/bin/env python3
"""A wrapper script for calling retrieval making functions.
All modules MUST have an add_arguments function
which adds the subcommand to the subparser.
"""

import argparse
import sys

from py_make_retrieval import make_ret_all


def main(args):
    """main function of cli"""
    args = _parse_args(args)
    make_ret_all.main(args)


def _parse_args(args):
    """function to parse arguments"""
    parser = argparse.ArgumentParser(description="MWRpy processing main wrapper.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["process"],
        default="process",
        help="Command to execute.",
    )
    group = parser.add_argument_group(title="Retrieval type")
    group.add_argument(
        "-r",
        "--ret",
        required=True,
        help="Retrieval type to be created, e.g. lwp",
        type=str,
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
