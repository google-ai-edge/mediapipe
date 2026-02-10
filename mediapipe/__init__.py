# Copyright 2019 - 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

try:
  import mediapipe.tasks.python as tasks
  from mediapipe.tasks.python.vision.core.image import Image
  from mediapipe.tasks.python.vision.core.image import ImageFormat
except Exception as e:  # pragma: no cover - optional tasks dependencies
  tasks = None
  Image = None
  ImageFormat = None
  warnings.warn(
      "MediaPipe tasks APIs could not be imported. Some functionality may be "
      f"unavailable. Original error: {e}",
      RuntimeWarning,
  )

try:
  from mediapipe.python import solutions
except Exception as e:  # pragma: no cover - optional solutions dependencies
  solutions = None
  warnings.warn(
      "MediaPipe solutions APIs could not be imported. Some functionality may "
      f"be unavailable. Original error: {e}",
      RuntimeWarning,
  )


__version__ = '0.10.32'