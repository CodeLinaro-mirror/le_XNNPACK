"""Generates an identifier from the files in the XNNPack project

This generates a fingerprint of the XNNPack library sources.
"""

import argparse
import hashlib
import os
import sys

parser = argparse.ArgumentParser(
    prog="XNNPackFingerprint",
    description=(
        "Generates a C source file that defeines a function that returns a"
        " fingerprint of the given XNNPack source files and writes it to the"
        " output."
    ),
)
parser.add_argument(
    "--output", required=True, action="store", help="Set the output"
)
parser.add_argument("inputs", nargs="+", help="The source files to use to generate the fingerprint.")

FILE_TEMPLATE = """// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Auto-generated file. Do not edit!
//   Generator: scripts/generate-build-identifier.py

#include <stdlib.h>

size_t xnn_get_build_identifier() {{
  return {}ull;
}}"""


def main(args) -> None:
  m = hashlib.sha256()
  for path in args.inputs:
    if any(path.endswith(ext) for ext in [".c", ".S", ".cc", ".h"]):
      with open(path, "rb") as f:
        m.update(f.read())
  with open(args.output, "w") as out:
    out.write(FILE_TEMPLATE.format(int.from_bytes(m.digest()[:8])))


if __name__ == "__main__":
  main(parser.parse_args())
