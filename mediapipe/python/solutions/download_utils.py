# Copyright 2021 The MediaPipe Authors.
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
"""MediaPipe Downloading utils."""

import os
import shutil
import urllib.request

_GCS_URL_PREFIX = 'https://storage.googleapis.com/mediapipe-assets/'


def download_oss_model(model_path: str):
  """Downloads the oss model from Google Cloud Storage if it doesn't exist in the package."""

  mp_root_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4])
  model_abspath = os.path.join(mp_root_path, model_path)
  if os.path.exists(model_abspath):
    return
  model_url = _GCS_URL_PREFIX + model_path.split('/')[-1]
  print('Downloading model to ' + model_abspath)
  with urllib.request.urlopen(model_url) as response, open(model_abspath,
                                                           'wb') as out_file:
    if response.code != 200:
      raise ConnectionError('Cannot download ' + model_path +
                            ' from Google Cloud Storage.')
    shutil.copyfileobj(response, out_file)
