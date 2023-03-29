#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D NR=2 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-x2-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D NR=4 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-x4-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D NR=8 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-x8-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D NR=2 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-x2-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D NR=4 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-x4-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D NR=8 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-x8-scalar-float.c &

################################## Unit tests #################################
tools/generate-packb-test.py --spec test/x32-zerob.yaml --output test/x32-zerob.cc &

wait
