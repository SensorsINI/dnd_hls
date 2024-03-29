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
"""Export quantizer package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .accumulator_factory import AccumulatorFactory
from .multiplier_factory import MultiplierFactory
from .multiplier_impl import IMultiplier, FloatingPointMultiplier, FixedPointMultiplier, Mux, AndGate, Adder, XorGate, Shifter
from .accumulator_impl import IAccumulator, FloatingPointAccumulator, FixedPointAccumulator
from .quantizer_impl import IQuantizer, QuantizedBits, Binary, QuantizedRelu, Ternary, FloatingPoint, PowerOfTwo, ReluPowerOfTwo
from .quantizer_factory import QuantizerFactory
from .qbn_factory import QBNFactory
from .fused_bn_factory import FusedBNFactory
from .merge_factory import MergeFactory
from .divider_factory import IDivider
from .subtractor_factory import ISubtractor
