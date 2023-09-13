# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
"""Face stylizer model constants."""

from mediapipe.model_maker.python.core.utils import file_util

# TODO: Move model files to GCS for downloading.
FACE_STYLIZER_ENCODER_MODEL_FILES = file_util.DownloadedFiles(
    'face_stylizer/encoder',
    'https://storage.googleapis.com/mediapipe-assets/face_stylizer_encoder.tar.gz',
    is_folder=True,
)
FACE_STYLIZER_DECODER_MODEL_FILES = file_util.DownloadedFiles(
    'face_stylizer/decoder',
    'https://storage.googleapis.com/mediapipe-assets/face_stylizer_decoder.tar.gz',
    is_folder=True,
)
FACE_STYLIZER_MAPPING_MODEL_FILES = file_util.DownloadedFiles(
    'face_stylizer/mapping',
    'https://storage.googleapis.com/mediapipe-assets/face_stylizer_mapping.tar.gz',
    is_folder=True,
)
FACE_STYLIZER_DISCRIMINATOR_MODEL_FILES = file_util.DownloadedFiles(
    'face_stylizer/discriminator',
    'https://storage.googleapis.com/mediapipe-assets/face_stylizer_discriminator.tar.gz',
    is_folder=True,
)
FACE_STYLIZER_W_FILES = file_util.DownloadedFiles(
    'face_stylizer/w_avg.npy',
    'https://storage.googleapis.com/mediapipe-assets/face_stylizer_w_avg.npy',
)

FACE_ALIGNER_TASK_FILES = file_util.DownloadedFiles(
    'face_stylizer/face_landmarker_v2.task',
    'https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task',
    is_folder=False,
)

# Dimension of the input style vector to the decoder
STYLE_DIM = 512
