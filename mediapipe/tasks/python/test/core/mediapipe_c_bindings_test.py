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

from absl.testing import absltest
from absl.testing import parameterized
from mediapipe.tasks.python.core import mediapipe_c_bindings

MpStatus = mediapipe_c_bindings.MpStatus


class MediapipeCBindingsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("cancelled", MpStatus.MP_CANCELLED, TimeoutError),
      ("unknown", MpStatus.MP_UNKNOWN, RuntimeError),
      (
          "invalid_argument",
          MpStatus.MP_INVALID_ARGUMENT,
          ValueError,
      ),
      (
          "deadline_exceeded",
          MpStatus.MP_DEADLINE_EXCEEDED,
          TimeoutError,
      ),
      ("not_found", MpStatus.MP_NOT_FOUND, FileNotFoundError),
      (
          "already_exists",
          MpStatus.MP_ALREADY_EXISTS,
          FileExistsError,
      ),
      (
          "permission_denied",
          MpStatus.MP_PERMISSION_DENIED,
          PermissionError,
      ),
      (
          "resource_exhausted",
          MpStatus.MP_RESOURCE_EXHAUSTED,
          RuntimeError,
      ),
      (
          "failed_precondition",
          MpStatus.MP_FAILED_PRECONDITION,
          RuntimeError,
      ),
      ("aborted", MpStatus.MP_ABORTED, RuntimeError),
      ("out_of_range", MpStatus.MP_OUT_OF_RANGE, IndexError),
      (
          "unimplemented",
          MpStatus.MP_UNIMPLEMENTED,
          NotImplementedError,
      ),
      ("internal", MpStatus.MP_INTERNAL, RuntimeError),
      (
          "unavailable",
          MpStatus.MP_UNAVAILABLE,
          ConnectionError,
      ),
      ("data_loss", MpStatus.MP_DATA_LOSS, RuntimeError),
      (
          "unauthenticated",
          MpStatus.MP_UNAUTHENTICATED,
          PermissionError,
      ),
  )
  def test_handle_status_raises_correct_errors(
      self, status, expected_exception
  ):
    with self.assertRaises(expected_exception):
      mediapipe_c_bindings.handle_status(status)

  def test_handle_status_ok(self):
    self.assertIsNone(mediapipe_c_bindings.handle_status(MpStatus.MP_OK))


if __name__ == "__main__":
  absltest.main()
