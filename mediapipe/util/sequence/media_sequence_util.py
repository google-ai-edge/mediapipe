"""Copyright 2019 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Utilities for creating MediaSequence functions.

A set of lightweight functions to simplify access to tensorflow SequenceExample
features and functions to create getters and setters for common types.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf


def add_functions_to_module(function_dict, module_dict=None):
  """Adds functions to another module.

  Args:
    function_dict: a list of names and function references. The functions will
      be accessible in the new module by the name.
    module_dict: The results of globals() for the current module or
      a_module.__dict__() call. Adding values to these dicts makes the functions
      accessible from the module. If not specified, globals() is used.
  """
  if module_dict is None:
    module_dict = globals()
  for name in function_dict:
    module_dict[name] = function_dict[name]


def merge_prefix(prefix, key):
  if prefix:
    return "/".join((prefix, key))
  else:
    return key


def has_context(key, sequence, prefix=""):
  return merge_prefix(prefix, key) in sequence.context.feature


def clear_context(key, sequence, prefix=""):
  del sequence.context.feature[merge_prefix(prefix, key)]


def set_context_float(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(prefix, key)].float_list.value[:] = (
      value,)


def get_context_float(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(
      prefix, key)].float_list.value[0]


def set_context_bytes(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].bytes_list.value[:] = (value,)


def get_context_bytes(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].bytes_list.value[0]


def set_context_int(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].int64_list.value[:] = (value,)


def get_context_int(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].int64_list.value[0]


def set_context_float_list(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].float_list.value[:] = value


def get_context_float_list(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].float_list.value


def set_context_bytes_list(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].bytes_list.value[:] = value


def get_context_bytes_list(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].bytes_list.value


def set_context_int_list(key, value, sequence, prefix=""):
  sequence.context.feature[merge_prefix(
      prefix, key)].int64_list.value[:] = value


def get_context_int_list(key, sequence, prefix=""):
  return sequence.context.feature[merge_prefix(prefix, key)].int64_list.value


def has_feature_list(key, sequence, prefix=""):
  return merge_prefix(prefix, key) in sequence.feature_lists.feature_list


def get_feature_list_size(key, sequence, prefix=""):
  if has_feature_list(merge_prefix(prefix, key), sequence):
    return len(sequence.feature_lists.feature_list[merge_prefix(
        prefix, key)].feature)
  else:
    return 0


def clear_feature_list(key, sequence, prefix=""):
  del sequence.feature_lists.feature_list[merge_prefix(prefix, key)]


def get_float_list_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].float_list.value


def get_int_list_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].int64_list.value


def get_bytes_list_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].bytes_list.value


def add_float_list(key, values, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).float_list.value[:] = values


def add_bytes_list(key, values, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).bytes_list.value[:] = values


def add_int_list(key, values, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).int64_list.value[:] = values


def get_float_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].float_list.value[0]


def get_int_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].int64_list.value[0]


def get_bytes_at(key, index, sequence, prefix=""):
  return sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature[
      index].bytes_list.value[0]


def add_float(key, value, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).float_list.value[:] = (value,)


def add_bytes(key, value, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).bytes_list.value[:] = (value,)


def add_int(key, value, sequence, prefix=""):
  sequence.feature_lists.feature_list[merge_prefix(prefix, key)].feature.add(
  ).int64_list.value[:] = (value,)


def create_bytes_list_context_feature(name, key, prefix="", module_dict=None):
  """Creates accessor functions for list of bytes features.

  The provided functions are has_${NAME}, get_${NAME}, set_${NAME} and
  clear_${NAME}.

  Example:
    example = tensorflow.train.SequenceExample()
    set_clip_label_string(["dog", "cat"], example)
    if has_clip_label_string(example):
      clip_label_string = get_clip_label_string(example)
      clear_clip_label_string(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)

  def _get(sequence_example, prefix=prefix):
    return get_context_bytes_list(key, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)

  def _set(value, sequence_example, prefix=prefix):
    set_context_bytes_list(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.VarLenFeature(tf.string)

  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_float_list_context_feature(name, key, prefix="", module_dict=None):
  """Creates accessor functions for list of floats features.

  The provided functions are has_${NAME}, get_${NAME}, set_${NAME} and
  clear_${NAME}.

  Example:
    example = tensorflow.train.SequenceExample()
    set_segment_label_confidence([0.47, 0.49], example)
    if has_segment_label_confidence(example):
      confidences = get_segment_label_confidence(example)
      clear_segment_label_confidence(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)

  def _get(sequence_example, prefix=prefix):
    return get_context_float_list(key, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)

  def _set(value, sequence_example, prefix=prefix):
    set_context_float_list(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.VarLenFeature(tf.float32)

  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_int_list_context_feature(name, key, prefix="", module_dict=None):
  """Creates accessor functions for list of int64 features.

  The provided functions are has_${NAME}, get_${NAME}, set_${NAME} and
  clear_${NAME}.

  Example:
    example = tensorflow.train.SequenceExample()
    set_clip_label_index([0, 1, 2, 3], example)
    if has_clip_label_index(example):
      clip_label_index = get_clip_label_index(example)
      clear_clip_label_index

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)

  def _get(sequence_example, prefix=prefix):
    return get_context_int_list(key, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)

  def _set(value, sequence_example, prefix=prefix):
    set_context_int_list(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.VarLenFeature(tf.int64)

  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_bytes_context_feature(name, key, prefix="", module_dict=None):
  """Creates accessor functions for single bytes features.

  The provided functions are has_${NAME}, get_${NAME}, set_${NAME} and
  clear_${NAME}.

  Example:
    example = tensorflow.train.SequenceExample()
    set_data_path("path_to_a_file", example)
    if has_data_path(example):
      path = get_data_path(example)
      clear_data_path(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)

  def _get(sequence_example, prefix=prefix):
    return get_context_bytes(key, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)

  def _set(value, sequence_example, prefix=prefix):
    set_context_bytes(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.FixedLenFeature((), tf.string)

  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_float_context_feature(name, key, prefix="", module_dict=None):
  """Creates accessor functions for single float features.

  The provided functions are has_${NAME}, get_${NAME}, set_${NAME} and
  clear_${NAME}.

  Example:
    example = tensorflow.train.SequenceExample()
    set_image_frame_rate(0.47, example)
    if has_image_frame_rate(example):
      image_frame_rate = get_image_frame_rate(example)
      clear_image_frame_rate(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)

  def _get(sequence_example, prefix=prefix):
    return get_context_float(key, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)

  def _set(value, sequence_example, prefix=prefix):
    set_context_float(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.FixedLenFeature((), tf.float32)

  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_int_context_feature(name, key, prefix="", module_dict=None):
  """Creates accessor functions for single int64 features.

  The provided functions are has_${NAME}, get_${NAME}, set_${NAME} and
  clear_${NAME}.

  Example:
    example = tensorflow.train.SequenceExample()
    set_clip_frames(47, example)
    if has_clip_frames(example):
      clip_frames = get_clip_frames(example)
      clear_clip_frames(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_context(key, sequence_example, prefix=prefix)

  def _get(sequence_example, prefix=prefix):
    return get_context_int(key, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_context(key, sequence_example, prefix=prefix)

  def _set(value, sequence_example, prefix=prefix):
    set_context_int(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.FixedLenFeature((), tf.int64)

  function_dict = {
      "has_" + name: _has,
      "get_" + name: _get,
      "clear_" + name: _clear,
      "set_" + name: _set,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_bytes_feature_list(name, key, prefix="", module_dict=None):
  """Creates accessor functions for bytes feature lists.

  The provided functions are has_${NAME}, get_${NAME}_size, get_${NAME}_at,
  clear_${NAME}, and add_${NAME}.

  example = tensorflow.train.SequenceExample()
  add_image_encoded(1000000, example)
  add_image_encoded(2000000, example)
  if has_image_encoded:
    for i in range(get_image_encoded_size(example):
      image_encoded = get_image_encoded_at(i, example)
    clear_image_encoded(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)

  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)

  def _get_at(index, sequence_example, prefix=prefix):
    return get_bytes_at(key, index, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)

  def _add(value, sequence_example, prefix=prefix):
    add_bytes(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.FixedLenSequenceFeature((), tf.string)

  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_float_feature_list(name, key, prefix="", module_dict=None):
  """Creates accessor functions for float feature lists.

  The provided functions are has_${NAME}, get_${NAME}_size, get_${NAME}_at,
  clear_${NAME}, and add_${NAME}.

  example = tensorflow.train.SequenceExample()
  add_confidence(0.47, example)
  add_confidence(0.49, example)
  if has_confidence:
    for i in range(get_confidence_size(example):
      confidence = get_confidence_at(i, example)
    clear_confidence(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)

  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)

  def _get_at(index, sequence_example, prefix=prefix):
    return get_float_at(key, index, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)

  def _add(value, sequence_example, prefix=prefix):
    add_float(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.FixedLenSequenceFeature((), tf.float32)

  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_int_feature_list(name, key, prefix="", module_dict=None):
  """Creates accessor functions for bytes feature lists.

  The provided functions are has_${NAME}, get_${NAME}_size, get_${NAME}_at,
  clear_${NAME}, and add_${NAME}.

  example = tensorflow.train.SequenceExample()
  add_image_timestamp(1000000, example)
  add_image_timestamp(2000000, example)
  if has_image_timestamp:
    for i in range(get_image_timestamp_size(example):
      timestamp = get_image_timestamp_at(i, example)
    clear_image_timestamp(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)

  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)

  def _get_at(index, sequence_example, prefix=prefix):
    return get_int_at(key, index, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)

  def _add(value, sequence_example, prefix=prefix):
    add_int(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.FixedLenSequenceFeature((), tf.int64)

  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_bytes_list_feature_list(name, key, prefix="", module_dict=None):
  """Creates accessor functions for list of bytes feature lists.

  The provided functions are has_${NAME}, get_${NAME}_size, get_${NAME}_at,
  clear_${NAME}, and add_${NAME}.

  example = tensorflow.train.SequenceExample()
  add_bbox_label_string(["dog", "cat"], example)
  add_bbox_label_string(["cat", "dog"], example)
  if has_bbox_label_string:
    for i in range(get_bbox_label_string_size(example):
      timestamp = get_bbox_label_string_at(i, example)
    clear_bbox_label_string(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)

  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)

  def _get_at(index, sequence_example, prefix=prefix):
    return get_bytes_list_at(key, index, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)

  def _add(value, sequence_example, prefix=prefix):
    add_bytes_list(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.VarLenFeature(tf.string)

  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_float_list_feature_list(name, key, prefix="", module_dict=None):
  """Creates accessor functions for list of float feature lists.

  The provided functions are has_${NAME}, get_${NAME}_size, get_${NAME}_at,
  clear_${NAME}, and add_${NAME}.

  example = tensorflow.train.SequenceExample()
  add_bbox_ymin([0.47, 0.49], example)
  add_bbox_ymin([0.49, 0.47], example)
  if has_bbox_ymin:
    for i in range(get_bbox_ymin_size(example):
      bbox_ymin = get_bbox_ymin_at(i, example)
    clear_bbox_ymin(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)

  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)

  def _get_at(index, sequence_example, prefix=prefix):
    return get_float_list_at(key, index, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)

  def _add(value, sequence_example, prefix=prefix):
    add_float_list(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.VarLenFeature(tf.float32)

  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)


def create_int_list_feature_list(name, key, prefix="", module_dict=None):
  """Creates accessor functions for list of int64 feature lists.

  The provided functions are has_${NAME}, get_${NAME}_size, get_${NAME}_at,
  clear_${NAME}, and add_${NAME}.

  example = tensorflow.train.SequenceExample()
  add_bbox_label_index([47, 49], example)
  add_bbox_label_index([49, 47], example)
  if has_bbox_label_index:
    for i in range(get_bbox_label_index_size(example):
      bbox_label_index = get_bbox_label_index_at(i, example)
    clear_bbox_label_index(example)

  Args:
    name: the name of the feature to use in function names.
    key: the key for this feature in the SequenceExample.
    prefix: a prefix to append to the key in the SequenceExample
    module_dict: adds the functions to the corresponding module dict.
  """
  def _has(sequence_example, prefix=prefix):
    return has_feature_list(key, sequence_example, prefix=prefix)

  def _get_size(sequence_example, prefix=prefix):
    return get_feature_list_size(key, sequence_example, prefix=prefix)

  def _get_at(index, sequence_example, prefix=prefix):
    return get_int_list_at(key, index, sequence_example, prefix=prefix)

  def _clear(sequence_example, prefix=prefix):
    clear_feature_list(key, sequence_example, prefix=prefix)

  def _add(value, sequence_example, prefix=prefix):
    add_int_list(key, value, sequence_example, prefix=prefix)

  def _get_key(prefix=prefix):
    return merge_prefix(prefix, key)

  def _get_default_parser():
    return tf.io.VarLenFeature(tf.int64)

  function_dict = {
      "has_" + name: _has,
      "get_" + name + "_size": _get_size,
      "get_" + name + "_at": _get_at,
      "clear_" + name: _clear,
      "add_" + name: _add,
      "get_" + name + "_key": _get_key,
      "get_" + name + "_default_parser": _get_default_parser,
  }
  add_functions_to_module(function_dict, module_dict)
