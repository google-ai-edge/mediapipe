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
"""MediaPipe image embedder task."""

import ctypes
import dataclasses
from typing import Callable, Optional

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import embedding_result_c as embedding_result_c_module
from mediapipe.tasks.python.components.utils import cosine_similarity
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

ImageEmbedderResult = embedding_result_module.EmbeddingResult
_BaseOptions = base_options_module.BaseOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher
_LiveStreamPacket = async_result_dispatcher.LiveStreamPacket


class _EmbedderOptionsC(ctypes.Structure):
  """C struct for embedder options."""

  _fields_ = [
      ('l2_normalize', ctypes.c_bool),
      ('quantize', ctypes.c_bool),
  ]


class ImageEmbedderOptionsC(ctypes.Structure):
  """The MediaPipe Tasks ImageEmbedderOptions CTypes struct."""

  _fields_ = [
      ('base_options', base_options_c_module.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('embedder_options', _EmbedderOptionsC),
      (
          'result_callback',
          ctypes.CFUNCTYPE(
              None,
              ctypes.c_int32,  # MpStatus
              ctypes.POINTER(embedding_result_c_module.EmbeddingResultC),
              ctypes.c_void_p,  # image
              ctypes.c_int64,  # timestamp_ms
          ),
      ),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpImageEmbedderCreate',
        (
            ctypes.POINTER(ImageEmbedderOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpImageEmbedderEmbedImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,  # image
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.POINTER(embedding_result_c_module.EmbeddingResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpImageEmbedderEmbedForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,  # image
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.c_int64,  # timestamp_ms
            ctypes.POINTER(embedding_result_c_module.EmbeddingResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpImageEmbedderEmbedAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,  # image
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.c_int64,  # timestamp_ms
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpImageEmbedderCloseResult',
        [ctypes.POINTER(embedding_result_c_module.EmbeddingResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpImageEmbedderClose',
        (ctypes.c_void_p,),
    ),
)

C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(embedding_result_c_module.EmbeddingResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


@dataclasses.dataclass
class ImageEmbedderOptions:
  """Options for the image embedder task.

  Attributes:
    base_options: Base options for the image embedder task.
    running_mode: The running mode of the task. Default to the image mode. Image
      embedder task has three running modes: 1) The image mode for embedding
      image on single image inputs. 2) The video mode for embedding image on the
      decoded frames of a video. 3) The live stream mode for embedding image on
      a live stream of input data, such as from camera.
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
      Use this option only if the model does not already contain a native
      L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
      L2 norm is thus achieved through TF Lite inference.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None
  result_callback: Optional[
      Callable[[ImageEmbedderResult, image_module.Image, int], None]
  ] = None

  _result_callback_c: Optional[
      Callable[
          [
              ctypes.c_int32,  # MpStatus
              ctypes.c_void_p,  # EmbeddingResultC
              ctypes.c_void_p,  # MpImage
              ctypes.c_int64,  # timestamp_ms
          ],
          None,
      ]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_ctypes(
      self, dispatcher: async_result_dispatcher.AsyncResultDispatcher
  ) -> ImageEmbedderOptionsC:
    """Generates an ImageEmbedderOptionsC object."""
    self._result_callback_c = dispatcher.wrap_callback(
        self.result_callback, C_TYPES_RESULT_CALLBACK
    )
    base_options_c = self.base_options.to_ctypes()
    embedder_options_c = _EmbedderOptionsC(
        l2_normalize=self.l2_normalize, quantize=self.quantize
    )
    return ImageEmbedderOptionsC(
        base_options=base_options_c,
        running_mode=self.running_mode.ctype,
        embedder_options=embedder_options_c,
        result_callback=self._result_callback_c,
    )


class ImageEmbedder:
  """Class that performs embedding extraction on images.

  The API expects a TFLite model with optional, but strongly recommended,
  TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  At least one output tensor with:
    (kTfLiteUInt8/kTfLiteFloat32)
    - `N` components corresponding to the `N` dimensions of the returned
      feature vector for this output layer.
    - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
  """

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p
  _dispatcher: async_result_dispatcher.AsyncResultDispatcher

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
      dispatcher: async_result_dispatcher.AsyncResultDispatcher,
  ):
    """Initializes a new ImageEmbedder instance.

    Args:
      lib: The dispatcher to use for all C function calls.
      handle: The C pointer to the underlying image embedder.
      dispatcher: The handler for async results.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageEmbedder':
    """Creates an `ImageEmbedder` object from a TensorFlow Lite model and the default `ImageEmbedderOptions`.

    Note that the created `ImageEmbedder` instance is in image mode, for
    embedding image on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageEmbedder` object that's created from the model file and the default
      `ImageEmbedderOptions`.

    Raises:
      ValueError: If failed to create `ImageEmbedder` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ImageEmbedderOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: ImageEmbedderOptions
  ) -> 'ImageEmbedder':
    """Creates the `ImageEmbedder` object from image embedder options.

    Args:
      options: Options for the image embedder task.

    Returns:
      `ImageEmbedder` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ImageEmbedder` object from
        `ImageEmbedderOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    running_mode_module.validate_running_mode(
        options.running_mode, options.result_callback
    )
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(
            embedding_result_c_module.EmbeddingResultC
        ),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> tuple[ImageEmbedderResult, image_module.Image, int]:
      """Converts an async C++ result to Python objects.

      Args:
        c_result_ptr: The pointer to the C result object.
        image_ptr: The pointer to the C input image.
        timestamp_ms: The timestamp of the input image in milliseconds.

      Returns:
       The connverted Python objects as a tuple.
      """
      c_result = c_result_ptr[0]
      py_result = embedding_result_module.EmbeddingResult.from_ctypes(c_result)
      py_image = image_module.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)
    ctypes_options = options.to_ctypes(dispatcher)

    embedder_handle = ctypes.c_void_p()
    lib.MpImageEmbedderCreate(
        ctypes.byref(ctypes_options), ctypes.byref(embedder_handle)
    )
    return cls(lib, embedder_handle, dispatcher=dispatcher)

  def embed(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ImageEmbedderResult:
    """Performs image embedding extraction on the provided MediaPipe Image.

     Extraction is performed on the region of interest specified by the `roi`
     argument if provided, or on the entire image otherwise.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      An embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image embedder failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = embedding_result_c_module.EmbeddingResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpImageEmbedderEmbedImage(
        self._handle,
        c_image,
        options_c,
        ctypes.byref(c_result),
    )
    py_result = embedding_result_module.EmbeddingResult.from_ctypes(c_result)
    self._lib.MpImageEmbedderCloseResult(ctypes.byref(c_result))
    return py_result

  def embed_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ImageEmbedderResult:
    """Performs image embedding extraction on the provided video frames.

    Extraction is performed on the region of interested specified by the `roi`
    argument if provided, or on the entire image otherwise.

    Only use this method when the ImageEmbedder is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      An embedding result object that contains a list of embeddings.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image embedder failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = embedding_result_c_module.EmbeddingResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpImageEmbedderEmbedForVideo(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
        ctypes.byref(c_result),
    )

    py_result = embedding_result_module.EmbeddingResult.from_ctypes(c_result)
    self._lib.MpImageEmbedderCloseResult(ctypes.byref(c_result))
    return py_result

  def embed_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to embedder.

    The results will be available via the "result_callback" provided in the
    ImageEmbedderOptions. Embedding extraction is performed on the region of
    interested specified by the `roi` argument if provided, or on the entire
    image otherwise.

    Only use this method when the ImageEmbedder is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `ImageEmbedderOptions`. The
    `embed_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, image embedder may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - An embedding result object that contains a list of embeddings.
      - The input image that the image embedder runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        embedder has already processed.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpImageEmbedderEmbedAsync(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
    )

  def close(self):
    """Closes the ImageEmbedder."""
    if not self._handle:
      return
    self._lib.MpImageEmbedderClose(self._handle)
    self._handle = None
    self._dispatcher.close()
    self._lib.close()

  @classmethod
  def cosine_similarity(
      cls,
      u: embedding_result_module.Embedding,
      v: embedding_result_module.Embedding,
  ) -> float:
    """Utility function to compute cosine similarity between two embedding entries.

    May return an InvalidArgumentError if e.g. the feature vectors are of
    different types (quantized vs. float), have different sizes, or have an
    L2-norm of 0.

    Args:
      u: An embedding entry.
      v: An embedding entry.

    Returns:
      The cosine similarity for the two embeddings.

    Raises:
      ValueError: May return an error if e.g. the feature vectors are of
        different types (quantized vs. float), have different sizes, or have
        an L2-norm of 0.
    """
    return cosine_similarity.cosine_similarity(u, v)

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager."""
    del exc_type, exc_value, traceback
    self.close()

  def __del__(self):
    self.close()
