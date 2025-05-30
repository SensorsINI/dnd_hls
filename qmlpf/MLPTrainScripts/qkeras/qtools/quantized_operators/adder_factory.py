## This file is part of https://github.com/SensorsINI/dnd_hls. 
## This intellectual property is licensed under the terms of the project license available at the root of the project.
# Copyright 2019 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""implement adder quantizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy

from absl import logging
from qkeras.qtools.quantized_operators import adder_impl
from qkeras.qtools.quantized_operators import quantizer_impl


class IAdder(abc.ABC):
  """abstract class for adder."""

  def __init__(self):
    self.adder_impl_table = [
        [
            adder_impl.FixedPointAdder,
            adder_impl.Po2FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FloatingPointAdder
        ],
        [
            adder_impl.Po2FixedPointAdder,
            adder_impl.Po2Adder,
            adder_impl.Po2FixedPointAdder,
            adder_impl.Po2FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FloatingPointAdder
        ],
        [
            adder_impl.FixedPointAdder,
            adder_impl.Po2FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FloatingPointAdder
        ],
        [
            adder_impl.FixedPointAdder,
            adder_impl.Po2FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FloatingPointAdder
        ],
        [
            adder_impl.FixedPointAdder,
            adder_impl.Po2FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FixedPointAdder,
            adder_impl.FloatingPointAdder
        ],
        [
            adder_impl.FloatingPointAdder,
            adder_impl.FloatingPointAdder,
            adder_impl.FloatingPointAdder,
            adder_impl.FloatingPointAdder,
            adder_impl.FloatingPointAdder,
            adder_impl.FloatingPointAdder
        ]
    ]

  def make_quantizer(self, quantizer_1: quantizer_impl.IQuantizer,
                     quantizer_2: quantizer_impl.IQuantizer):
    """make adder quantizer."""

    local_quantizer_1 = copy.deepcopy(quantizer_1)
    local_quantizer_2 = copy.deepcopy(quantizer_2)

    mode1 = local_quantizer_1.mode
    mode2 = local_quantizer_2.mode

    adder_impl_class = self.adder_impl_table[mode1][mode2]
    logging.debug(
        "qbn adder implemented as class %s",
        adder_impl_class.implemented_as())

    return adder_impl_class(
        local_quantizer_1,
        local_quantizer_2
    )
