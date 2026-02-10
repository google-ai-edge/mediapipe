# Copyright 2022 The MediaPipe Authors.
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

"""MediaPipe Tasks API."""

import importlib
import warnings


def _safe_import(name: str):
  try:
    return importlib.import_module(name)
  except Exception as e:  # pragma: no cover - optional modules
    warnings.warn(
        f"MediaPipe Tasks submodule '{name}' could not be imported: {e}",
        RuntimeWarning,
    )
    return None


audio = _safe_import(__name__ + ".audio")
components = _safe_import(__name__ + ".components")
core = _safe_import(__name__ + ".core")
genai = _safe_import(__name__ + ".genai")
text = _safe_import(__name__ + ".text")
vision = _safe_import(__name__ + ".vision")

if core is not None:
  BaseOptions = core.base_options.BaseOptions
else:
  BaseOptions = None

# Remove unnecessary modules to avoid duplication in API docs.
if core is not None:
  del core

