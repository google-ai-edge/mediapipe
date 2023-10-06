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
# ==============================================================================
"""Utility methods for creating the model asset bundles."""

from typing import Dict
import zipfile

# Alignment that ensures that all uncompressed files in the model bundle file
# are aligned relative to the start of the file. This lets the files be
# accessed directly via mmap.
_ALIGNMENT = 4


class AlignZipFile(zipfile.ZipFile):
  """ZipFile that stores uncompressed files at particular alignment."""

  def __init__(self, *args, alignment: int, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    assert alignment > 0
    self._alignment = alignment

  def _writecheck(self, zinfo: zipfile.ZipInfo) -> None:
    # Aligned the uncompressed files.
    if zinfo.compress_type == zipfile.ZIP_STORED:
      offset = self.fp.tell()
      header_length = len(zinfo.FileHeader())
      padding_length = (
          self._alignment - (offset + header_length) % self._alignment
      )
      if padding_length:
        offset += padding_length
        self.fp.write(b"\x00" * padding_length)
        assert self.fp.tell() == offset
        zinfo.header_offset = offset
    else:
      raise ValueError(
          "Only support the uncompressed file (compress_type =="
          " zipfile.ZIP_STORED) in zip. The current file compress type is "
          + str(zinfo.compress_type)
      )
    super()._writecheck(zinfo)


def create_model_asset_bundle(
    input_models: Dict[str, bytes], output_path: str
) -> None:
  """Creates the model asset bundle.

  Args:
    input_models: A dict of input models with key as the model file name and
      value as the model content.
    output_path: The output file path to save the model asset bundle.
  """
  if not input_models or len(input_models) < 2:
    raise ValueError("Needs at least two input models for model asset bundle.")

  with AlignZipFile(output_path, mode="w", alignment=_ALIGNMENT) as zf:
    for file_name, file_buffer in input_models.items():
      zf.writestr(file_name, file_buffer)
