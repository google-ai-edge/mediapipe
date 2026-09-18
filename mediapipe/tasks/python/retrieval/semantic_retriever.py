# pytype: skip-file
# Copyright 2026 The MediaPipe Authors.
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
"""MediaPipe semantic retriever task."""

import ctypes
import dataclasses
from typing import Any, Dict, List, Optional, Union

from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.retrieval import universal_embedder

_UniversalEmbedder = universal_embedder.UniversalEmbedder


_DEFAULT_EMBEDDING_DIMENSION = 768


class TaskPartKind:
  """Part kind enum matching C/C++ definitions."""

  TEXT = 0
  IMAGE = 1
  AUDIO = 2


class ChunkingMode:
  """Mode for text chunking matching C/C++ definitions."""

  CHARACTER = 0
  WORD = 1


class _MpKeyValuePairC(ctypes.Structure):
  """C struct for MpKeyValuePair."""

  _fields_ = [
      ('key', ctypes.c_char_p),
      ('value', ctypes.c_char_p),
  ]


class _MpSemanticRetrieverOptionsC(ctypes.Structure):
  """C struct for MpSemanticRetrieverOptions."""

  _fields_ = [
      ('database_path', ctypes.c_char_p),
      ('embedding_dimension', ctypes.c_int),
      ('embedder', ctypes.c_void_p),
      ('text_embedder', ctypes.c_void_p),
      ('image_embedder', ctypes.c_void_p),
      ('chunk_size', ctypes.c_int),
      ('chunk_overlap', ctypes.c_int),
      ('chunking_mode', ctypes.c_int),
  ]


class _MpRetrievalRecordC(ctypes.Structure):
  """C struct for MpRetrievalRecord."""

  _fields_ = [
      ('id', ctypes.c_char_p),
      ('text', ctypes.c_char_p),
      ('score', ctypes.c_float),
      ('metadata', ctypes.POINTER(_MpKeyValuePairC)),
      ('metadata_count', ctypes.c_int),
  ]


class _MpRetrievalResultC(ctypes.Structure):
  """C struct for MpRetrievalResult."""

  _fields_ = [
      ('records', ctypes.POINTER(_MpRetrievalRecordC)),
      ('records_count', ctypes.c_int),
  ]


class _MpRecordIdsResultC(ctypes.Structure):
  """C struct for MpRecordIdsResult."""

  _fields_ = [
      ('ids', ctypes.POINTER(ctypes.c_char_p)),
      ('ids_count', ctypes.c_int),
  ]


class _MpTextPartC(ctypes.Structure):
  _fields_ = [('text', ctypes.c_char_p)]


class _MpImagePartC(ctypes.Structure):
  _fields_ = [
      ('image_bytes', ctypes.POINTER(ctypes.c_uint8)),
      ('image_bytes_size', ctypes.c_int),
      ('file_path', ctypes.c_char_p),
  ]


class _MpAudioPartC(ctypes.Structure):
  _fields_ = [
      ('audio_data', ctypes.POINTER(ctypes.c_float)),
      ('audio_data_size', ctypes.c_int),
      ('audio_path', ctypes.c_char_p),
  ]


class _MpTaskPartC(ctypes.Structure):
  _fields_ = [
      ('kind', ctypes.c_int),
      ('text_part', _MpTextPartC),
      ('image_part', _MpImagePartC),
      ('audio_part', _MpAudioPartC),
  ]


@dataclasses.dataclass
class RetrievalRecord:
  """Retrieval record representing a single matched database document."""

  id: str
  text: str
  score: float
  metadata: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RetrievalResult:
  """Retrieval result containing matching documents."""

  records: List[RetrievalRecord]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverCreate',
        [
            ctypes.POINTER(_MpSemanticRetrieverOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverInsertDocument',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpKeyValuePairC),
            ctypes.c_int,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverInsertImage',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(_MpKeyValuePairC),
            ctypes.c_int,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverInsertAudio',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(_MpKeyValuePairC),
            ctypes.c_int,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverInsertContent',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpTaskPartC),
            ctypes.c_int,
            ctypes.POINTER(_MpKeyValuePairC),
            ctypes.c_int,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverRetrieve',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpTaskPartC),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(_MpRetrievalResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverRetrieveWithMetadataFilter',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpTaskPartC),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_MpKeyValuePairC),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(_MpRetrievalResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverDelete',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverDeleteWithMetadataFilter',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpKeyValuePairC),
            ctypes.c_int,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverGetAllRecordIds',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpRecordIdsResultC),
        ],
    ),
    mediapipe_c_utils.CFunction(
        'MpSemanticRetrieverCloseRecordIdsResult',
        [ctypes.POINTER(_MpRecordIdsResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverDeleteAll',
        [
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CFunction(
        'MpSemanticRetrieverCloseResult',
        [ctypes.POINTER(_MpRetrievalResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpSemanticRetrieverClose',
        [
            ctypes.c_void_p,
        ],
    ),
)


@dataclasses.dataclass
class SemanticRetrieverOptions:
  """Options for the semantic retriever task."""

  embedder: _UniversalEmbedder
  embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION
  database_path: Optional[str] = None
  chunk_size: int = 512
  chunk_overlap: int = 100
  chunking_mode: int = ChunkingMode.CHARACTER

  def to_ctypes(self) -> _MpSemanticRetrieverOptionsC:
    """Generates a ctypes MpSemanticRetrieverOptionsC object."""
    return _MpSemanticRetrieverOptionsC(
        database_path=self.database_path.encode('utf-8')
        if self.database_path
        else None,
        embedding_dimension=self.embedding_dimension,
        embedder=self.embedder._embedder_handle,  # pylint: disable=protected-access
        text_embedder=None,
        image_embedder=None,
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
        chunking_mode=self.chunking_mode,
    )


class SemanticRetriever:
  """Class that manages a vector index and retrieves relevant media records."""

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ) -> None:
    self._lib = lib
    self._retriever_handle = handle

  @classmethod
  def create_from_options(
      cls, options: SemanticRetrieverOptions
  ) -> 'SemanticRetriever':
    """Creates the `SemanticRetriever` object from options."""
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)
    ctypes_options = options.to_ctypes()
    retriever_handle = ctypes.c_void_p()
    lib.MpSemanticRetrieverCreate(
        ctypes.byref(ctypes_options), ctypes.byref(retriever_handle)
    )
    return SemanticRetriever(lib=lib, handle=retriever_handle)

  def _convert_metadata(
      self, metadata: Dict[str, str]
  ) -> List[Optional[_MpKeyValuePairC]]:
    """Helper to convert Python dict to ctypes array of key-value pairs."""
    arr_type = _MpKeyValuePairC * len(metadata)
    arr = arr_type()
    for i, (key, value) in enumerate(metadata.items()):
      arr[i].key = key.encode('utf-8')
      arr[i].value = value.encode('utf-8')
    return arr

  def _to_retrieval_record(self, rec_c: _MpRetrievalRecordC) -> RetrievalRecord:
    """Converts a ctypes _MpRetrievalRecordC to a Python RetrievalRecord."""
    meta_dict = {}
    if rec_c.metadata and rec_c.metadata_count > 0:
      for j in range(rec_c.metadata_count):
        kv = rec_c.metadata[j]
        meta_dict[kv.key.decode('utf-8')] = kv.value.decode('utf-8')
    return RetrievalRecord(
        id=rec_c.id.decode('utf-8'),
        text=rec_c.text.decode('utf-8'),
        score=rec_c.score,
        metadata=meta_dict,
    )

  def insert_document(
      self,
      doc_id: str,
      text: str,
      metadata: Optional[Dict[str, str]] = None,
  ) -> None:
    """Inserts text document with optional metadata."""
    arr = self._convert_metadata(metadata) if metadata else None
    self._lib.MpSemanticRetrieverInsertDocument(
        self._retriever_handle,
        doc_id.encode('utf-8'),
        text.encode('utf-8'),
        arr,
        len(metadata) if metadata else 0,
    )

  def insert_image(
      self,
      image_id: str,
      image_path: str,
      image_bytes: Optional[bytes] = None,
      metadata: Optional[Dict[str, str]] = None,
  ) -> None:
    """Inserts an image into the vector store.

    If `image_bytes` is provided, the image is loaded from those raw bytes
    directly. Otherwise, if `image_bytes` is not passed, the image is
    automatically loaded and decoded from `image_path` on-demand.

    Args:
      image_id: The unique identifier for the image record.
      image_path: The file path or URI for the image. Must be specified.
      image_bytes: Optional raw encoded image bytes (e.g. JPEG, PNG).
      metadata: Optional dictionary of key-value metadata to associate.
    """
    arr = self._convert_metadata(metadata) if metadata else None

    if image_bytes is not None:
      bytes_arr = ctypes.cast(image_bytes, ctypes.POINTER(ctypes.c_uint8))
      bytes_len = len(image_bytes)
    else:
      bytes_arr = None
      bytes_len = 0

    self._lib.MpSemanticRetrieverInsertImage(
        self._retriever_handle,
        image_id.encode('utf-8'),
        bytes_arr,
        bytes_len,
        image_path.encode('utf-8'),
        arr,
        len(metadata) if metadata else 0,
    )

  def insert_audio(
      self,
      audio_id: str,
      audio_path: str,
      audio_data: Optional[List[float]] = None,
      metadata: Optional[Dict[str, str]] = None,
  ) -> None:
    """Inserts an audio record into the vector store.

    If `audio_data` is provided, the audio is loaded from those raw PCM samples
    directly. Otherwise, if `audio_data` is not passed, the audio is
    automatically loaded and decoded from `audio_path` on-demand.

    Args:
      audio_id: The unique identifier for the audio record.
      audio_path: The file path to the WAV file. Must be specified.
      audio_data: Optional list of raw mono PCM float samples.
      metadata: Optional dictionary of key-value metadata to associate.
    """
    arr = self._convert_metadata(metadata) if metadata else None

    if audio_data is not None and len(audio_data) > 0:
      audio_arr = (ctypes.c_float * len(audio_data))()
      audio_arr[:] = audio_data
      audio_len = len(audio_data)
    else:
      audio_arr = None
      audio_len = 0

    self._lib.MpSemanticRetrieverInsertAudio(
        self._retriever_handle,
        audio_id.encode('utf-8'),
        audio_arr,
        audio_len,
        audio_path.encode('utf-8'),
        arr,
        len(metadata) if metadata else 0,
    )

  def insert_content(
      self,
      record_id: str,
      parts: List[Dict[str, Any]],
      metadata: Optional[Dict[str, str]] = None,
  ) -> None:
    """Inserts multi-modal parts with optional metadata.

    Args:
      record_id: The unique identifier for the content record.
      parts: A list of part dictionaries. Each part dictionary should match the
        structure `{'kind': TaskPartKind, 'text': ..., 'image_bytes': ...,
        'file_path': ..., 'audio_data': ..., 'audio_path': ...}`.
      metadata: Optional dictionary of key-value metadata to associate.
    """
    arr = self._convert_metadata(metadata) if metadata else None
    arr_ref = ctypes.byref(arr) if arr else None

    parts_arr_type = _MpTaskPartC * len(parts)
    parts_arr = parts_arr_type()

    # Keep references to byte arrays to prevent garbage collection
    refs = []

    for i, part in enumerate(parts):
      kind = part.get('kind', TaskPartKind.TEXT)
      parts_arr[i].kind = kind
      if kind == TaskPartKind.TEXT:
        text = part.get('text', '')
        encoded_text = text.encode('utf-8')
        refs.append(encoded_text)
        parts_arr[i].text_part.text = encoded_text
      elif kind == TaskPartKind.IMAGE:
        img_bytes = part.get('image_bytes')
        file_path = part.get('file_path', '')
        if img_bytes is not None:
          bytes_arr = ctypes.cast(img_bytes, ctypes.POINTER(ctypes.c_uint8))
          refs.append(img_bytes)
          parts_arr[i].image_part.image_bytes = bytes_arr
          parts_arr[i].image_part.image_bytes_size = len(img_bytes)
        else:
          parts_arr[i].image_part.image_bytes = None
          parts_arr[i].image_part.image_bytes_size = 0
        encoded_path = file_path.encode('utf-8') if file_path else None
        refs.append(encoded_path)
        parts_arr[i].image_part.file_path = encoded_path
      elif kind == TaskPartKind.AUDIO:
        audio_data = part.get('audio_data')
        audio_path = part.get('audio_path', '')
        if audio_data is not None:
          audio_arr = (ctypes.c_float * len(audio_data))()
          audio_arr[:] = audio_data
          refs.append(audio_arr)
          parts_arr[i].audio_part.audio_data = audio_arr
          parts_arr[i].audio_part.audio_data_size = len(audio_data)
        else:
          parts_arr[i].audio_part.audio_data = None
          parts_arr[i].audio_part.audio_data_size = 0
        encoded_path = audio_path.encode('utf-8') if audio_path else None
        refs.append(encoded_path)
        parts_arr[i].audio_part.audio_path = encoded_path

    self._lib.MpSemanticRetrieverInsertContent(
        self._retriever_handle,
        record_id.encode('utf-8'),
        parts_arr,
        len(parts),
        arr_ref,
        len(metadata) if metadata else 0,
    )

  def retrieve(
      self,
      query: Union[str, List[Dict[str, Any]]],
      limit: int,
      min_similarity: float = 0.5,
      metadata_filter: Optional[Dict[str, str]] = None,
  ) -> RetrievalResult:
    """Retrieves nearest matching documents from the vector store.

    Args:
      query: Either a plain text query string, or a list of query part
        dictionaries for multimodal retrieval. Each part dictionary should match
        the structure `{'kind': TaskPartKind, 'text': ..., 'image_bytes': ...,
        'file_path': ..., 'audio_data': ..., 'audio_path': ...}`.
      limit: The maximum number of nearest records to return.
      min_similarity: The minimum similarity score threshold.
      metadata_filter: Optional dictionary of key-value pairs used to filter
        candidate records.

    Returns:
      A `RetrievalResult` containing matching documents.
    """
    result_c = _MpRetrievalResultC()
    if isinstance(query, str):
      query = [{'kind': TaskPartKind.TEXT, 'text': query}]

    parts_arr_type = _MpTaskPartC * len(query)
    parts_arr = parts_arr_type()

    # Keep references to byte arrays to prevent garbage collection
    refs = []

    for i, part in enumerate(query):
      kind = part.get('kind', TaskPartKind.TEXT)
      parts_arr[i].kind = kind
      if kind == TaskPartKind.TEXT:
        text = part.get('text', '')
        encoded_text = text.encode('utf-8')
        refs.append(encoded_text)
        parts_arr[i].text_part.text = encoded_text
      elif kind == TaskPartKind.IMAGE:
        img_bytes = part.get('image_bytes')
        file_path = part.get('file_path', '')
        if img_bytes is not None:
          bytes_arr = ctypes.cast(img_bytes, ctypes.POINTER(ctypes.c_uint8))
          refs.append(img_bytes)
          parts_arr[i].image_part.image_bytes = bytes_arr
          parts_arr[i].image_part.image_bytes_size = len(img_bytes)
        else:
          parts_arr[i].image_part.image_bytes = None
          parts_arr[i].image_part.image_bytes_size = 0
        encoded_path = file_path.encode('utf-8') if file_path else None
        refs.append(encoded_path)
        parts_arr[i].image_part.file_path = encoded_path
      elif kind == TaskPartKind.AUDIO:
        audio_data = part.get('audio_data')
        audio_path = part.get('audio_path', '')
        if audio_data is not None:
          audio_arr = (ctypes.c_float * len(audio_data))()
          audio_arr[:] = audio_data
          refs.append(audio_arr)
          parts_arr[i].audio_part.audio_data = audio_arr
          parts_arr[i].audio_part.audio_data_size = len(audio_data)
        else:
          parts_arr[i].audio_part.audio_data = None
          parts_arr[i].audio_part.audio_data_size = 0
        encoded_path = audio_path.encode('utf-8') if audio_path else None
        refs.append(encoded_path)
        parts_arr[i].audio_part.audio_path = encoded_path

    if metadata_filter:
      filter_arr = self._convert_metadata(metadata_filter)
      self._lib.MpSemanticRetrieverRetrieveWithMetadataFilter(
          self._retriever_handle,
          parts_arr,
          len(query),
          limit,
          filter_arr,
          len(metadata_filter),
          min_similarity,
          ctypes.byref(result_c),
      )
    else:
      self._lib.MpSemanticRetrieverRetrieve(
          self._retriever_handle,
          parts_arr,
          len(query),
          limit,
          min_similarity,
          ctypes.byref(result_c),
      )

    records = []
    for i in range(result_c.records_count):
      records.append(self._to_retrieval_record(result_c.records[i]))

    self._lib.MpSemanticRetrieverCloseResult(ctypes.byref(result_c))
    return RetrievalResult(records=records)

  def delete_record(self, record_id: str) -> None:
    """Deletes record with the specified ID."""
    self._lib.MpSemanticRetrieverDelete(
        self._retriever_handle,
        record_id.encode('utf-8'),
    )

  def delete_with_metadata_filter(
      self, metadata_filter: Dict[str, str]
  ) -> None:
    """Deletes records matching the specified metadata filter."""
    if not metadata_filter:
      return
    arr = self._convert_metadata(metadata_filter)
    self._lib.MpSemanticRetrieverDeleteWithMetadataFilter(
        self._retriever_handle,
        arr,
        len(metadata_filter),
    )

  def get_all_record_ids(self) -> List[str]:
    """Returns list of all active record IDs."""
    ids_res = _MpRecordIdsResultC()
    self._lib.MpSemanticRetrieverGetAllRecordIds(
        self._retriever_handle,
        ctypes.byref(ids_res),
    )

    ids = []
    for i in range(ids_res.ids_count):
      ids.append(ids_res.ids[i].decode('utf-8'))

    self._lib.MpSemanticRetrieverCloseRecordIdsResult(ctypes.byref(ids_res))
    return ids

  def delete_all(self) -> None:
    """Deletes all records from database."""
    self._lib.MpSemanticRetrieverDeleteAll(self._retriever_handle)

  def close(self) -> None:
    """Shuts down and releases the retriever."""
    if not self._retriever_handle:
      return
    self._lib.MpSemanticRetrieverClose(self._retriever_handle)
    self._retriever_handle = None
    self._lib.close()

  def __enter__(self) -> 'SemanticRetriever':
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()
