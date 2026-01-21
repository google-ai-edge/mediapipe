# Copyright 2025 The MediaPipe Authors.
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

"""C types for ClassifierOptions."""

import ctypes

import mediapipe.tasks.python.components.processors.classifier_options as classifier_options_module


class ClassifierOptionsC(ctypes.Structure):
  _fields_ = [
      ("display_names_locale", ctypes.c_char_p),
      ("max_results", ctypes.c_int),
      ("score_threshold", ctypes.c_float),
      ("category_allowlist", ctypes.POINTER(ctypes.c_char_p)),
      ("category_allowlist_count", ctypes.c_uint32),
      ("category_denylist", ctypes.POINTER(ctypes.c_char_p)),
      ("category_denylist_count", ctypes.c_uint32),
  ]


def convert_to_classifier_options_c(
    src: classifier_options_module.ClassifierOptions,
) -> ClassifierOptionsC:
  """Converts a Python ClassifierOptions object to a ClassifierOptionsC object."""
  options = ClassifierOptionsC()

  if src.display_names_locale is not None:
    options.display_names_locale = src.display_names_locale.encode("utf-8")
  else:
    options.display_names_locale = None

  if src.max_results is not None:
    options.max_results = src.max_results
  else:
    options.max_results = -1

  if src.score_threshold is not None:
    options.score_threshold = src.score_threshold
  else:
    options.score_threshold = 0.0

  # Handle category_allowlist
  if src.category_allowlist:
    bytes_allowlist = [s.encode("utf-8") for s in src.category_allowlist]
    allowlist_array_type = ctypes.c_char_p * len(bytes_allowlist)
    allowlist_array = allowlist_array_type(*bytes_allowlist)
    options.category_allowlist = ctypes.cast(
        allowlist_array, ctypes.POINTER(ctypes.c_char_p)
    )
    options.category_allowlist_count = len(bytes_allowlist)
  else:
    options.category_allowlist = None
    options.category_allowlist_count = 0

  # Handle category_denylist
  if src.category_denylist:
    bytes_denylist = [s.encode("utf-8") for s in src.category_denylist]
    denylist_array_type = ctypes.c_char_p * len(bytes_denylist)
    denylist_array = denylist_array_type(*bytes_denylist)
    options.category_denylist = ctypes.cast(
        denylist_array, ctypes.POINTER(ctypes.c_char_p)
    )
    options.category_denylist_count = len(bytes_denylist)
  else:
    options.category_denylist = None
    options.category_denylist_count = 0

  return options
