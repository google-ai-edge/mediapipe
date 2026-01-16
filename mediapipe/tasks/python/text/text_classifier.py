# Copyright 2022 The MediaPipe Authors.
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
"""MediaPipe text classifier task."""

import ctypes
import dataclasses
from typing import List, Optional
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.components.containers import classification_result_c as classification_result_c_module
from mediapipe.tasks.python.components.processors import classifier_options as classifier_options_module
from mediapipe.tasks.python.components.processors import classifier_options_c as classifier_options_c_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher

_BaseOptions = base_options_module.BaseOptions
Category = category_module.Category
Classifications = classification_result_module.Classifications
TextClassifierResult = classification_result_module.ClassificationResult
ClassifierOptions = classifier_options_module.ClassifierOptions


class TextClassifierOptionsC(ctypes.Structure):
  _fields_ = [
      ("base_options", base_options_c_module.BaseOptionsC),
      ("classifier_options", classifier_options_c_module.ClassifierOptionsC),
  ]


@dataclasses.dataclass
class TextClassifierOptions:
  """Options for the text classifier task.

  Attributes:
    base_options: Base options for the text classifier task.
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

  def to_ctypes(self) -> TextClassifierOptionsC:
    """Generates a ctypes TextClassifierOptionsC."""
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

    c_options = TextClassifierOptionsC()
    c_options.base_options = base_options_c
    c_options.classifier_options = classifier_options_c
    return c_options


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        "MpTextClassifierCreate",
        [
            ctypes.POINTER(TextClassifierOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextClassifierClassify",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(
                classification_result_c_module.ClassificationResultC
            ),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextClassifierClose",
        [
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CFunction(
        "MpTextClassifierCloseResult",
        [ctypes.POINTER(classification_result_c_module.ClassificationResultC)],
        None,
    ),
)


class TextClassifier:
  """Class that performs classification on text.

  This API expects a TFLite model with (optional) TFLite Model Metadata that
  contains the mandatory (described below) input tensors, output tensor,
  and the optional (but recommended) category labels as AssociatedFiles with
  type
  TENSOR_AXIS_LABELS per output classification tensor. Metadata is required for
  models with int32 input tensors because it contains the input process unit
  for the model's Tokenizer. No metadata is required for models with string
  input tensors.

  Input tensors:
    (kTfLiteInt32)
    - 3 input tensors of size `[batch_size x bert_max_seq_len]` representing
      the input ids, segment ids, and mask ids
    - or 1 input tensor of size `[batch_size x max_seq_len]` representing the
      input ids
    or (kTfLiteString)
    - 1 input tensor that is shapeless or has shape [1] containing the input
      string
  At least one output tensor with:
    (kTfLiteFloat32/kBool)
    - `[1 x N]` array with `N` represents the number of categories.
    - optional (but recommended) category labels as AssociatedFiles with type
      TENSOR_AXIS_LABELS, containing one label per line. The first such
      AssociatedFile (if any) is used to fill the `category_name` field of the
      results. The `display_name` field is filled from the AssociatedFile (if
      any) whose locale matches the `display_names_locale` field of the
      `TextClassifierOptions` used at creation time ("en" by default, i.e.
      English). If none of these are available, only the `index` field of the
      results will be filled.
  """
  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ):
    self._lib = lib
    self._classifier_handle = handle

  @classmethod
  def create_from_model_path(cls, model_path: str) -> "TextClassifier":
    """Creates an `TextClassifier` object from a TensorFlow Lite model and the default `TextClassifierOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `TextClassifier` object that's created from the model file and the
      default `TextClassifierOptions`.

    Raises:
      ValueError: If failed to create `TextClassifier` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    return cls.create_from_options(
        TextClassifierOptions(
            base_options=_BaseOptions(model_asset_path=model_path)
        )
    )

  @classmethod
  def create_from_options(
      cls, options: TextClassifierOptions
  ) -> "TextClassifier":
    """Creates the `TextClassifier` object from text classifier options.

    Args:
      options: Options for the text classifier task.

    Returns:
      `TextClassifier` object that's created from `options`.

    Raises:
      ValueError: If failed to create `TextClassifier` object from
        `TextClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    ctypes_options = options.to_ctypes()

    classifier_handle = ctypes.c_void_p()
    lib.MpTextClassifierCreate(
        ctypes.byref(ctypes_options),
        ctypes.byref(classifier_handle),
    )
    return TextClassifier(lib=lib, handle=classifier_handle)

  def classify(self, text: str) -> TextClassifierResult:
    """Performs classification on the input `text`.

    Args:
      text: The input text.

    Returns:
      A `TextClassifierResult` object that contains a list of text
      classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If text classification failed to run.
    """
    ctypes_result = classification_result_c_module.ClassificationResultC()

    self._lib.MpTextClassifierClassify(
        self._classifier_handle,
        text.encode("utf-8"),
        ctypes.byref(ctypes_result),
    )
    python_result = TextClassifierResult.from_ctypes(ctypes_result)
    self._lib.MpTextClassifierCloseResult(ctypes.byref(ctypes_result))
    return python_result

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._classifier_handle:
      self._lib.MpTextClassifierClose(self._classifier_handle)
      self._classifier_handle = None
      self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager.

    Raises:
      RuntimeError: If the MediaPipe TextClassifier task failed to close.
    """
    self.close()

  def __del__(self):
    self.close()
