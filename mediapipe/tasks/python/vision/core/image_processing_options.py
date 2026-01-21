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
"""MediaPipe vision options for image processing."""

import dataclasses
from typing import Optional

from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.components.containers import rect_c as rect_c_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module


@dataclasses.dataclass
class ImageProcessingOptions:
  """Options for image processing.

  If both region-of-interest and rotation are specified, the crop around the
  region-of-interest is extracted first, then the specified rotation is applied
   to the crop.

  Attributes:
    region_of_interest: The optional region-of-interest to crop from the image.
      If not specified, the full image is used. Coordinates must be in [0,1]
      with 'left' < 'right' and 'top' < 'bottom'.
    rotation_degrees: The rotation to apply to the image (or cropped
      region-of-interest), in degrees clockwise. The rotation must be a multiple
      (positive or negative) of 90°.
  """
  region_of_interest: Optional[rect_module.RectF] = None
  rotation_degrees: int = 0

  @doc_controls.do_not_generate_docs
  def to_ctypes(
      self,
  ) -> image_processing_options_c_module.ImageProcessingOptionsC:
    """Generates a ImageProcessingOptionsC object."""
    return image_processing_options_c_module.ImageProcessingOptionsC(
        has_region_of_interest=self.region_of_interest is not None,
        region_of_interest=(
            self.region_of_interest.to_ctypes()
            if self.region_of_interest
            else rect_c_module.RectFC()
        ),
        rotation_degrees=self.rotation_degrees,
    )
