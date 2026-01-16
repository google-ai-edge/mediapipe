# Copyright 2023 The MediaPipe Authors.
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
"""MediaPipe language detector task."""

import ctypes
import dataclasses
from typing import List, Optional
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.components.processors import classifier_options as classifier_options_module
from mediapipe.tasks.python.components.processors import classifier_options_c as classifier_options_c_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings as mediapipe_c_bindings_module
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher

_BaseOptions = base_options_module.BaseOptions
Category = category_module.Category
Classifications = classification_result_module.Classifications
ClassifierOptions = classifier_options_module.ClassifierOptions


@dataclasses.dataclass
class LanguageDetectorResult:

  @dataclasses.dataclass
  class Detection:
    """A language code and its probability."""

    # An i18n language / locale code, e.g. "en" for English, "uz" for Uzbek,
    # "ja"-Latn for Japanese (romaji).
    language_code: str
    probability: float

  detections: List[Detection]


class LanguageDetectorPredictionC(ctypes.Structure):
  """A language code and its probability."""

  _fields_ = [
      ("language_code", ctypes.c_char_p),
      ("probability", ctypes.c_float),
  ]


class LanguageDetectorResultC(ctypes.Structure):
  """Language detector output."""

  _fields_ = [
      ("predictions", ctypes.POINTER(LanguageDetectorPredictionC)),
      ("predictions_count", ctypes.c_uint32),
  ]


def _convert_to_python_language_detector_result(
    c_result: LanguageDetectorResultC,
) -> LanguageDetectorResult:
  """Converts a C LanguageDetectorResult to a Python LanguageDetectorResult."""
  py_result = LanguageDetectorResult(detections=[])
  for i in range(c_result.predictions_count):
    c_prediction = c_result.predictions[i]
    py_prediction = LanguageDetectorResult.Detection(
        language_code=c_prediction.language_code.decode("utf-8"),
        probability=c_prediction.probability,
    )
    py_result.detections.append(py_prediction)
  return py_result


class LanguageDetectorOptionsC(ctypes.Structure):
  _fields_ = [
      ("base_options", base_options_c_module.BaseOptionsC),
      ("classifier_options", classifier_options_c_module.ClassifierOptionsC),
  ]


@dataclasses.dataclass
class LanguageDetectorOptions:
  """Options for the language detector task.

  Attributes:
    base_options: Base options for the language detector task.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty,
      classification results whose category name is not in this set will be
      filtered out. Duplicate or unknown category names are ignored. Mutually
      exclusive with `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, classification
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
  """
  base_options: _BaseOptions
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None

  def to_ctypes(self) -> LanguageDetectorOptionsC:
    """Generates a ctypes LanguageDetectorOptionsC."""
    base_options_c = self.base_options.to_ctypes()
    classifier_options_c = (
        classifier_options_c_module.convert_to_classifier_options_c(
            classifier_options_module.ClassifierOptions(
                display_names_locale=self.display_names_locale,
                max_results=self.max_results,
                score_threshold=self.score_threshold,
                category_allowlist=self.category_allowlist,
                category_denylist=self.category_denylist,
            )
        )
    )

    c_options = LanguageDetectorOptionsC()
    c_options.base_options = base_options_c
    c_options.classifier_options = classifier_options_c
    return c_options


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        "MpLanguageDetectorCreate",
        (
            ctypes.POINTER(LanguageDetectorOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpLanguageDetectorDetect",
        (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(LanguageDetectorResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpLanguageDetectorClose",
        (
            ctypes.c_void_p,
        ),
    ),
    mediapipe_c_utils.CFunction(
        "MpLanguageDetectorCloseResult",
        [ctypes.POINTER(LanguageDetectorResultC)],
        None,
    ),
)


class LanguageDetector:
  """Class that predicts the language of an input text.

  This API expects a TFLite model with TFLite Model Metadata that contains the
  mandatory (described below) input tensors, output tensor, and the language
  codes in an AssociatedFile.

  Input tensors:
    (kTfLiteString)
    - 1 input tensor that is scalar or has shape [1] containing the input
      string.
  Output tensor:
    (kTfLiteFloat32)
    - 1 output tensor of shape`[1 x N]` where `N` is the number of languages.
  """
  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self, lib: serial_dispatcher.SerialDispatcher, handle: ctypes.c_void_p
  ):
    self._lib = lib
    self._detector_handle = handle

  @classmethod
  def create_from_model_path(cls, model_path: str) -> "LanguageDetector":
    """Creates an `LanguageDetector` object from a TensorFlow Lite model and the default `LanguageDetectorOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `LanguageDetector` object that's created from the model file and the
      default `LanguageDetectorOptions`.

    Raises:
      ValueError: If failed to create `LanguageDetector` object from the
      provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    return cls.create_from_options(
        LanguageDetectorOptions(
            base_options=_BaseOptions(model_asset_path=model_path)
        )
    )

  @classmethod
  def create_from_options(
      cls, options: LanguageDetectorOptions
  ) -> "LanguageDetector":
    """Creates the `LanguageDetector` object from language detector options.

    Args:
      options: Options for the language detector task.

    Returns:
      `LanguageDetector` object that's created from `options`.

    Raises:
      ValueError: If failed to create `LanguageDetector` object from
        `LanguageDetectorOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings_module.load_shared_library(_CTYPES_SIGNATURES)

    ctypes_options = options.to_ctypes()

    detector_handle = ctypes.c_void_p()
    lib.MpLanguageDetectorCreate(
        ctypes.byref(ctypes_options),
        ctypes.byref(detector_handle),
    )
    return LanguageDetector(lib=lib, handle=detector_handle)

  def detect(self, text: str) -> LanguageDetectorResult:
    """Predicts the language of the input `text`.

    Args:
      text: The input text.

    Returns:
      A `LanguageDetectorResult` object that contains a list of languages and
      scores.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If language detection failed to run.
    """
    ctypes_result = LanguageDetectorResultC()

    self._lib.MpLanguageDetectorDetect(
        self._detector_handle,
        text.encode("utf-8"),
        ctypes.byref(ctypes_result),
    )
    python_result = _convert_to_python_language_detector_result(ctypes_result)
    self._lib.MpLanguageDetectorCloseResult(ctypes.byref(ctypes_result))
    return python_result

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if not self._detector_handle:
      return
    self._lib.MpLanguageDetectorClose(self._detector_handle)
    self._detector_handle = None
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager.

    Raises:
      RuntimeError: If the MediaPipe LanguageDetector task failed to close.
    """
    self.close()

  def __del__(self):
    self.close()
