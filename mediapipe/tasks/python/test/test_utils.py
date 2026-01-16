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
"""Test util for MediaPipe Tasks."""

import difflib
import os

from absl import flags
import six

from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import image_frame as image_frame_module

FLAGS = flags.FLAGS
_Image = image_module.Image
_ImageFormat = image_frame_module.ImageFormat
_RGB_CHANNELS = 3


def test_srcdir():
  """Returns the path where to look for test data files."""
  if "test_srcdir" in flags.FLAGS:
    return flags.FLAGS["test_srcdir"].value
  elif "TEST_SRCDIR" in os.environ:
    return os.environ["TEST_SRCDIR"]
  else:
    raise RuntimeError("Missing TEST_SRCDIR environment.")


def get_test_data_path(file_or_dirname_path: str) -> str:
  """Returns full test data path."""
  for directory, subdirs, files in os.walk(test_srcdir()):
    for f in subdirs + files:
      path = os.path.join(directory, f)
      if path.endswith(file_or_dirname_path):
        return path
  raise ValueError(
      "No %s in test directory: %s." % (file_or_dirname_path, test_srcdir())
  )


def create_calibration_file(
    file_dir: str,
    file_name: str = "score_calibration.txt",
    content: str = "1.0,2.0,3.0,4.0",
) -> str:
  """Creates the calibration file."""
  calibration_file = os.path.join(file_dir, file_name)
  with open(calibration_file, mode="w") as file:
    file.write(content)
  return calibration_file


def assert_proto_equals(
    self, a, b, check_initialized=True, normalize_numbers=True, msg=None
):
  """assert_proto_equals() is useful for unit tests.

  It produces much more helpful output than assertEqual() for proto2 messages.
  Fails with a useful error if a and b aren't equal. Comparison of repeated
  fields matches the semantics of unittest.TestCase.assertEqual(), ie order and
  extra duplicates fields matter.

  This is a fork of https://github.com/tensorflow/tensorflow/blob/
  master/tensorflow/python/util/protobuf/compare.py#L73. We use slightly
  different rounding cutoffs to support Mac usage.

  Args:
    self: absltest.testing.parameterized.TestCase
    a: proto2 PB instance, or text string representing one.
    b: proto2 PB instance -- message.Message or subclass thereof.
    check_initialized: boolean, whether to fail if either a or b isn't
      initialized.
    normalize_numbers: boolean, whether to normalize types and precision of
      numbers before comparison.
    msg: if specified, is used as the error message on failure.
  """
  pool = descriptor_pool.Default()
  if isinstance(a, six.string_types):
    a = text_format.Parse(a, b.__class__(), descriptor_pool=pool)

  for pb in a, b:
    if check_initialized:
      errors = pb.FindInitializationErrors()
      if errors:
        self.fail("Initialization errors: %s\n%s" % (errors, pb))
    if normalize_numbers:
      _normalize_number_fields(pb)

  a_str = text_format.MessageToString(a, descriptor_pool=pool)
  b_str = text_format.MessageToString(b, descriptor_pool=pool)

  # Some Python versions would perform regular diff instead of multi-line
  # diff if string is longer than 2**16. We substitute this behavior
  # with a call to unified_diff instead to have easier-to-read diffs.
  # For context, see: https://bugs.python.org/issue11763.
  if len(a_str) < 2**16 and len(b_str) < 2**16:
    self.assertMultiLineEqual(a_str, b_str, msg=msg)
  else:
    diff = "".join(
        difflib.unified_diff(a_str.splitlines(True), b_str.splitlines(True))
    )
    if diff:
      self.fail("%s :\n%s" % (msg, diff))


def _normalize_number_fields(pb):
  """Normalizes types and precisions of number fields in a protocol buffer.

  Due to subtleties in the python protocol buffer implementation, it is possible
  for values to have different types and precision depending on whether they
  were set and retrieved directly or deserialized from a protobuf. This function
  normalizes integer values to ints and longs based on width, 32-bit floats to
  five digits of precision to account for python always storing them as 64-bit,
  and ensures doubles are floating point for when they're set to integers.
  Modifies pb in place. Recurses into nested objects. https://github.com/tensorf
  low/tensorflow/blob/master/tensorflow/python/util/protobuf/compare.py#L118

  Args:
    pb: proto2 message.

  Returns:
    the given pb, modified in place.
  """
  for desc, values in pb.ListFields():
    is_repeated = True
    if desc.label != descriptor.FieldDescriptor.LABEL_REPEATED:
      is_repeated = False
      values = [values]

    normalized_values = None

    # We force 32-bit values to int and 64-bit values to long to make
    # alternate implementations where the distinction is more significant
    # (e.g. the C++ implementation) simpler.
    if desc.type in (
        descriptor.FieldDescriptor.TYPE_INT64,
        descriptor.FieldDescriptor.TYPE_UINT64,
        descriptor.FieldDescriptor.TYPE_SINT64,
    ):
      normalized_values = [int(x) for x in values]
    elif desc.type in (
        descriptor.FieldDescriptor.TYPE_INT32,
        descriptor.FieldDescriptor.TYPE_UINT32,
        descriptor.FieldDescriptor.TYPE_SINT32,
        descriptor.FieldDescriptor.TYPE_ENUM,
    ):
      normalized_values = [int(x) for x in values]
    elif desc.type == descriptor.FieldDescriptor.TYPE_FLOAT:
      normalized_values = [round(x, 4) for x in values]
    elif desc.type == descriptor.FieldDescriptor.TYPE_DOUBLE:
      normalized_values = [round(float(x), 6) for x in values]

    if normalized_values is not None:
      if is_repeated:
        pb.ClearField(desc.name)
        getattr(pb, desc.name).extend(normalized_values)
      else:
        setattr(pb, desc.name, normalized_values[0])

    if (
        desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE
        or desc.type == descriptor.FieldDescriptor.TYPE_GROUP
    ):
      if (
          desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE
          and desc.message_type.has_options
          and desc.message_type.GetOptions().map_entry
      ):
        # This is a map, only recurse if the values have a message type.
        if (
            desc.message_type.fields_by_number[2].type
            == descriptor.FieldDescriptor.TYPE_MESSAGE
        ):
          for v in six.itervalues(values):
            _normalize_number_fields(v)
      else:
        for v in values:
          # recursive step
          _normalize_number_fields(v)

  return pb
