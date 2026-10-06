# Copyright 2026 The MediaPipe Authors.
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
"""MediaPipe interactive segmenter task."""

from __future__ import annotations

import ctypes
import dataclasses
import enum
from typing import List

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_module

_BaseOptions = base_options_module.BaseOptions
_CStatusFunction = mediapipe_c_utils.CStatusFunction


class InteractiveSegmenterOptionsC(ctypes.Structure):
  """The MediaPipe Tasks InteractiveSegmenterOptions CTypes struct."""

  _fields_ = [
      ('base_options', base_options_c_module.MpBaseOptionsC),
  ]


class MpStrokePointC(ctypes.Structure):
  """The MediaPipe Tasks MpStrokePoint CTypes struct."""

  _fields_ = [
      ('x', ctypes.c_float),
      ('y', ctypes.c_float),
  ]


class MpStrokeC(ctypes.Structure):
  """The MediaPipe Tasks MpStroke CTypes struct."""

  _fields_ = [
      ('brush_mode', ctypes.c_int),
      ('points', ctypes.POINTER(MpStrokePointC)),
      ('points_count', ctypes.c_uint32),
      ('is_completed', ctypes.c_bool),
  ]


class MpStrokesC(ctypes.Structure):
  """The MediaPipe Tasks MpStrokes CTypes struct."""

  _fields_ = [
      ('strokes', ctypes.POINTER(MpStrokeC)),
      ('strokes_count', ctypes.c_uint32),
  ]


_CTYPES_SIGNATURES = (
    _CStatusFunction(
        'MpInteractiveSegmenterCreate',
        (
            ctypes.POINTER(InteractiveSegmenterOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    _CStatusFunction(
        'MpInteractiveSegmenterSetImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
        ),
    ),
    _CStatusFunction(
        'MpInteractiveSegmenterSegment',
        (
            ctypes.c_void_p,
            ctypes.POINTER(MpStrokesC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    _CStatusFunction(
        'MpInteractiveSegmenterClose',
        (ctypes.c_void_p,),
    ),
)


class BrushMode(enum.IntEnum):
  """The brush mode for interactive segmentation."""
  UNSPECIFIED = 0
  # Includes area into segmentation output.
  POSITIVE = 1
  # Excludes area from segmentation output.
  NEGATIVE = 2
  # Includes selected object into segmentation output.
  LASSO = 3


@dataclasses.dataclass
class StrokePoint:
  """A single 2D point in a stroke.

  Attributes:
    x: The normalized x coordinate of the point.
    y: The normalized y coordinate of the point.
  """

  x: float
  y: float


@dataclasses.dataclass
class Stroke:
  """A stroke representing user interaction for segmentation.

  Attributes:
    brush_mode: The brush mode of the stroke.
    points: A list of points making up the stroke.
    is_completed: Whether the stroke is completed.
  """

  brush_mode: BrushMode
  points: List[StrokePoint]
  is_completed: bool


@dataclasses.dataclass
class InteractiveSegmenterOptions:
  """Options for the interactive segmenter task.

  Attributes:
    base_options: Base options for the interactive segmenter task.
  """

  base_options: _BaseOptions

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> InteractiveSegmenterOptionsC:
    """Generates an InteractiveSegmenterOptionsC ctypes struct."""
    return InteractiveSegmenterOptionsC(
        base_options=self.base_options.to_ctypes(),
    )


class InteractiveSegmenter:
  """A task to perform interactive segmentation on images."""

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ):
    """Initializes the instance."""
    self._lib = lib
    self._handle = handle

  @classmethod
  def create_from_options(
      cls, options: InteractiveSegmenterOptions
  ) -> InteractiveSegmenter:
    """Creates the `InteractiveSegmenter` object from `options`.

    Args:
      options: Options for the interactive segmenter task.

    Returns:
      `InteractiveSegmenter` object.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)
    ctypes_options = options.to_ctypes()

    segmenter_handle = ctypes.c_void_p()
    lib.MpInteractiveSegmenterCreate(
        ctypes.byref(ctypes_options), ctypes.byref(segmenter_handle)
    )
    return cls(lib, segmenter_handle)

  def set_image(self, image: image_module.Image) -> None:
    """Sets the image to be segmented."""
    c_image = image._image_ptr  # pylint: disable=protected-access
    self._lib.MpInteractiveSegmenterSetImage(self._handle, c_image)

  def segment(self, strokes: List[Stroke]) -> image_module.Image:
    """Performs segmentation on the previously set image."""
    c_strokes_array = (MpStrokeC * len(strokes))()

    # We must keep a reference to the python lists of points to prevent the
    # underlying C arrays from being garbage collected before the C function
    # finishes execution.
    c_stroke_points_arrays = []

    for i, stroke in enumerate(strokes):
      c_points = (MpStrokePointC * len(stroke.points))()
      for j, p in enumerate(stroke.points):
        c_points[j] = MpStrokePointC(x=p.x, y=p.y)
      c_stroke_points_arrays.append(c_points)

      c_strokes_array[i] = MpStrokeC(
          brush_mode=stroke.brush_mode.value,
          points=ctypes.cast(c_points, ctypes.POINTER(MpStrokePointC)),
          points_count=len(stroke.points),
          is_completed=stroke.is_completed,
      )

    c_strokes = MpStrokesC(
        strokes=ctypes.cast(c_strokes_array, ctypes.POINTER(MpStrokeC)),
        strokes_count=len(strokes),
    )

    c_mask_ptr = ctypes.c_void_p()
    self._lib.MpInteractiveSegmenterSegment(
        self._handle,
        ctypes.byref(c_strokes),
        ctypes.byref(c_mask_ptr),
    )

    mask = image_module.Image.create_from_ctypes(c_mask_ptr)
    # Free the mask pointer returned by the C API since create_from_ctypes
    # creates a new Image wrapper sharing the underlying pixel data.
    image_module.Image.free_ctypes(c_mask_ptr)

    return mask

  def close(self):
    """Closes the InteractiveSegmenter."""
    if not self._handle:
      return
    self._lib.MpInteractiveSegmenterClose(self._handle)
    self._handle = None
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager."""
    del exc_type, exc_value, traceback
    self.close()

  def __del__(self):
    self.close()
