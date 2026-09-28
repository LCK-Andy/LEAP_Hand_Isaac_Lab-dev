# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Simulator-free released cube-generator filename semantics and fixed variant ordering."""

import numpy as np
from itertools import product


def original_cube_catalog(randomize=True, small=True, big=True, sticks=True):
    """Reproduce generator filename overwrites, sorted multiplicities and fixed shuffle."""
    files = {}
    groups = [([100], 1.0, 1.0, False, False)]
    if small:
        groups.append(([100, 50, 66, 75, 90, 110, 125, 150, 175, 200, 250, 300], 1.0, 2.5, False, False))
    if big:
        groups.append(([100, 125, 150, 200, 250, 300, 350], 2.5, 15.0, True, False))
    if sticks:
        groups.append(([100, 50, 75, 200, 300, 400, 500, 600], 2.5, 6.0, True, True))
    for scales, minimum, maximum, reject_thin, reject_short in groups:
        index = 0
        for xyz in product(scales, repeat=3):
            volume = xyz[0] * xyz[1] * xyz[2] / 1000000
            lo, middle, hi = sorted(xyz)
            if not minimum <= volume <= maximum:
                continue
            if reject_thin and lo * 3 <= middle:
                continue
            if reject_short and (hi <= lo * 3 or hi <= middle * 3):
                continue
            filename = f"{index:03d}_cube_{xyz[0]}_{xyz[1]}_{xyz[2]}.urdf"
            files[filename] = tuple(value / 100 for value in xyz)
            index += 1
    ordered = [(name, files[name]) for name in sorted(files)]
    if not randomize:
        ordered = ordered[:1]
    np.random.default_rng(42).shuffle(ordered)
    return ordered
