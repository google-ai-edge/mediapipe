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
"""Utilities for files."""

import dataclasses
import os
import pathlib
import shutil
import tarfile
import tempfile
import requests

# resources dependency


_TEMPDIR_FOLDER = 'model_maker'


@dataclasses.dataclass
class DownloadedFiles:
  """File(s) that are downloaded from a url into a local directory.

  If `is_folder` is True:
    1. `path` should be a folder
    2. `url` should point to a .tar.gz file which contains a single folder at
      the root level.

  Attributes:
    path: Relative path in local directory.
    url: GCS url to download the file(s).
    is_folder: Whether the path and url represents a folder.
  """

  path: str
  url: str
  is_folder: bool = False

  def get_path(self) -> str:
    """Gets the path of files saved in a local directory.

    If the path doesn't exist, this method will download the file(s) from the
    provided url. The path is not cleaned up so it can be reused for subsequent
    calls to the same path.
    Folders are expected to be zipped in a .tar.gz file which will be extracted
    into self.path in the local directory.

    Raises:
      RuntimeError: If the extracted folder does not have a singular root
        directory.

    Returns:
      The absolute path to the downloaded file(s)
    """
    tmpdir = tempfile.gettempdir()
    absolute_path = pathlib.Path(
        os.path.join(tmpdir, _TEMPDIR_FOLDER, self.path)
    )
    if not absolute_path.exists():
      print(f'Downloading {self.url} to {absolute_path}')
      r = requests.get(self.url, allow_redirects=True)
      if self.is_folder:
        # Use tempf to store the downloaded .tar.gz file
        tempf = tempfile.NamedTemporaryFile(suffix='.tar.gz', mode='wb')
        tempf.write(r.content)
        tarf = tarfile.open(tempf.name)
        # Use tmpdir to store the extracted contents of the .tar.gz file
        with tempfile.TemporaryDirectory() as tmpdir:
          tarf.extractall(tmpdir)
          tarf.close()
          tempf.close()
          subdirs = os.listdir(tmpdir)
          # Make sure tmpdir only has one subdirectory
          if len(subdirs) > 1 or not os.path.isdir(
              os.path.join(tmpdir, subdirs[0])
          ):
            raise RuntimeError(
                f"Extracted folder from {self.url} doesn't contain a "
                f'single root directory: {subdirs}'
            )
          # Create the parent dir of absolute_path and copy the contents of the
          # top level dir in the .tar.gz file into absolute_path.
          pathlib.Path.mkdir(absolute_path.parent, parents=True, exist_ok=True)
          shutil.copytree(os.path.join(tmpdir, subdirs[0]), absolute_path)
      else:
        pathlib.Path.mkdir(absolute_path.parent, parents=True, exist_ok=True)
        with open(absolute_path, 'wb') as f:
          f.write(r.content)
    return str(absolute_path)


# TODO Remove after text_classifier supports downloading on demand.
def get_absolute_path(file_path: str) -> str:
  """Gets the absolute path of a file in the model_maker directory.

  Args:
    file_path: The path to a file relative to the `mediapipe` dir

  Returns:
   The full path of the file
  """
  # Extract the file path before and including 'model_maker' as the
  # `mm_base_dir`. By joining it with the `path` after 'model_maker/', it
  # yields to the absolute path of the model files directory. We must join
  # on 'model_maker' because in the pypi package, the 'model_maker' directory
  # is renamed to 'mediapipe_model_maker'. So we have to join on model_maker
  # to ensure that the `mm_base_dir` path includes the renamed
  # 'mediapipe_model_maker' directory.
  cwd = os.path.dirname(__file__)
  cwd_stop_idx = cwd.rfind('model_maker') + len('model_maker')
  mm_base_dir = cwd[:cwd_stop_idx]
  file_path_start_idx = file_path.find('model_maker') + len('model_maker') + 1
  mm_relative_path = file_path[file_path_start_idx:]
  absolute_path = os.path.join(mm_base_dir, mm_relative_path)
  return absolute_path
