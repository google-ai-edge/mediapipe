# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""MediaPipe text task base api."""

from mediapipe.framework import calculator_pb2
from mediapipe.python._framework_bindings import task_runner
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_TaskRunner = task_runner.TaskRunner


class BaseTextTaskApi(object):
  """The base class of the user-facing mediapipe text task api classes."""

  def __init__(self,
               graph_config: calculator_pb2.CalculatorGraphConfig) -> None:
    """Initializes the `BaseVisionTaskApi` object.

    Args:
      graph_config: The mediapipe text task graph config proto.
    """
    self._runner = _TaskRunner.create(graph_config)

  def close(self) -> None:
    """Shuts down the mediapipe text task instance.

    Raises:
      RuntimeError: If the mediapipe text task failed to close.
    """
    self._runner.close()

  @doc_controls.do_not_generate_docs
  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  @doc_controls.do_not_generate_docs
  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the mediapipe text task instance on exit of the context manager.

    Raises:
      RuntimeError: If the mediapipe text task failed to close.
    """
    self.close()
