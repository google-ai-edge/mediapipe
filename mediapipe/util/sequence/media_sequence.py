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


This script defines a large number of getters and setters for storing
multimedia, such as video or audio, and related machine learning data in
tf.train.SequenceExamples. These getters and setters simplify sharing
data by enforcing common patterns for storing data in SequenceExample
key-value pairs.

The constants, macros, and functions are organized into 6 groups: clip
metadata, clip label related, segment related, bounding-box related, image
related, feature list related, and keyframe related. The following examples
will walk through common task structures, but the relevant data to store can
vary by task.

The clip metadata group is generally data about the media and stored in the
SequenceExample.context. Specifying the metadata enables media pipelines,
such as MediaPipe, to retrieve that data. Typically, set_clip_data_path,
set_clip_start_timestamp, and set_clip_end_timestamp define which data to use
without storing the data itself. Example:
  tensorflow.train.SequenceExample sequence
  set_clip_data_path("/relative/path/to/data.mp4", sequence)
  set_clip_start_timestamp(0, sequence)
  set_clip_end_timestamp(10000000, sequence)  # 10 seconds in microseconds.

The clip label group adds labels that apply to the entire media clip. To
annotate that a video clip has a particular label, set the clip metadata
above and also set the set_clip_label_index and set_clip_label_string. Most
training pipelines will only use the label index or string, but we recommend
storing both to improve readability while maintaining ease of use.
Example:
  set_clip_label_string(("run", "jump"), sequence)
  set_Clip_label_index((35, 47), sequence)

The segment group is generally data about time spans within the media clip
and stored in the SequenceExample.context. In this code, continuous lengths
of media are called clips, and each clip may have subregions of interest that
are called segments. To annotate that a video clip has time spans with labels
set the clip metadata above and use the functions set_segment_start_timestamp,
set_segment_end_timestamp, set_segment_label_index, and
set_segment_label_string. Most training pipelines will only use the label index
or string, but we recommend storing both to improve readability while
maintaining ease of use. By listing segments as times, the frame rate or other
properties can change without affecting the labels.
Example:
  set_segment_start_timestamp((500000, 1000000), sequence)  # in microseconds
  set_segment_end_timestamp((2000000, 6000000), sequence)
  set_segment_label_index((35, 47), sequence)
  set_segment_label_string(("run", "jump"), sequence)

The bounding box group is useful for identifying spatio-temporal annotations
for detection, tracking, or action recognition. The exact keys that are
needed can vary by task, but to annotate a video clip for detection set the
clip metadata above and use repeatedly call add_bbox, add_bbox_timestamp,
add_bbox_label_index, and add_bbox_label_string. Most training pipelines will
only use the label index or string, but we recommend storing both to improve
readability while maintaining ease of use. Because bounding boxes are
assigned to timepoints in a video, changing the image frame rate can can
change the alignment. The media_sequence.h's ReconcileMetadata function can
align bounding boxes to the nearest image.

The image group is useful for storing data as sequential 2D arrays, typically
encoded as bytes. Images can be RGB images stored as JPEG, discrete masks
stored as PNG, or some other format. Parameters that are static over time are
set in the context using set_image_width, set_image_height, set_image_format,
etc. The series of frames and timestamps are then added with add_image_encoded
and
add_image_timestamp. For discrete masks, the class or instance indices can be
mapped to labels or classes using
set_class_segmentation_class_label_{index,string} and
set_instance_segmentation_object_class_index.

The feature list group is useful for storing audio and extracted features,
such as per-frame embeddings. SequenceExamples only store lists of floats per
timestep, so the dimensions are stored in the context to enable reshaping.
For example, set_feature_dimensions and repeatedly calling add_feature_floats
and add_feature_timestamp adds per-frame embeddings. The feature methods also
support audio features.

Macros for common patterns are created in media_sequence_util.py and are used
here extensively. Because these macros are formulaic, I will only include a
usage example here in the code rather than repeating documentation for every
instance. This header defines additional functions to simplify working with
MediaPipe types.

Each msu.create_{TYPE}_context_feature takes a NAME and a KEY. It provides
setters and getters for SequenceExamples and stores a single value under KEY
in the context field. The provided functions are has_${NAME}, get_${NAME},
set_${Name}, and clear_${NAME}.
Eg.
  tf.train.SequenceExample example
  set_data_path("data_path", example)
  if has_data_path(example):
     data_path = get_data_path(example)
     clear_data_path(example)

Each msu.create_{TYPE}_list_context_feature takes a NAME and a KEY. It provides
setters and getters for SequenceExamples and stores a sequence of values
under KEY in the context field. The provided functions are has_${NAME},
get_${NAME}, set_${Name}, clear_${NAME}, get_${NAME}_at, and add_${NAME}.
Eg.
  tf.train.SequenceExample example
  set_clip_label_string(("run", "jump"), example)
  if has_clip_label_string(example):
     values = get_clip_label_string(example)
     clear_clip_label_string(example)

Each msu.create_{TYPE}_feature_list takes a NAME and a KEY. It provides setters
and getters for SequenceExamples and stores a single value in each feature field
under KEY of the feature_lists field. The provided functions are has_${NAME},
get_${NAME}, clear_${NAME}, get_${NAME}_size, get_${NAME}_at, and add_${NAME}.
  tf.train.SequenceExample example
  add_image_timestamp(1000000, example)
  add_image_timestamp(2000000, example)
  if has_image_timestamp(example):
    for i in range(get_image_timestamp_size()):
      timestamp = get_image_timestamp_at(example, i)
    clear_image_timestamp(example)

Each VECTOR_{TYPE}_FEATURE_LIST takes a NAME and a KEY. It provides setters
and getters for SequenceExamples and stores a sequence of values in each
feature field under KEY of the feature_lists field. The provided functions
are Has${NAME}, Get${NAME}, Clear${NAME}, Get${NAME}Size, Get${NAME}At, and
Add${NAME}.
  tf.train.SequenceExample example
  add_bbox_label_string(("run", "jump"), example)
  add_bbox_label_string(("run", "fall"), example)
  if has_bbox_label_string(example):
    for i in range(get_bbox_label_string_size(example)):
      labels = get_bbox_label_string_at(example, i)
    clear_bbox_label_string(example)

As described in media_sequence_util.h, each of these functions can take an
additional string prefix argument as their first argument. The prefix can
be fixed with a new NAME by using functools.partial. Prefixes are used to
identify common storage patterns (e.g. storing an image along with the height
and width) under different names (e.g. storing a left and right image in a
stereo pair.) An example creating functions such as
add_left_image_encoded that adds a string under the key "LEFT/image/encoded"
 add_left_image_encoded = msu.function_with_default(add_image_encoded, "LEFT")
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from mediapipe.util.sequence import media_sequence_util
msu = media_sequence_util

_HAS_DYNAMIC_ATTRIBUTES = True

##################################  METADATA  #################################
# A unique identifier for each example.
EXAMPLE_ID_KEY = "example/id"
# The name o fthe data set, including the version.
EXAMPLE_DATASET_NAME_KEY = "example/dataset_name"
# String flags or attributes for this example within a data set.
EXAMPLE_DATASET_FLAG_STRING_KEY = "example/dataset/flag/string"
# The relative path to the data on disk from some root directory.
CLIP_DATA_PATH_KEY = "clip/data_path"
# Any identifier for the media beyond the data path.
CLIP_MEDIA_ID_KEY = "clip/media_id"
# Yet another alternative identifier.
ALTERNATIVE_CLIP_MEDIA_ID_KEY = "clip/alternative_media_id"
# The encoded bytes for storing media directly in the SequenceExample.
CLIP_ENCODED_MEDIA_BYTES_KEY = "clip/encoded_media_bytes"
# The start time for the encoded media if not preserved during encoding.
CLIP_ENCODED_MEDIA_START_TIMESTAMP_KEY = "clip/encoded_media_start_timestamp"
# The start time, in microseconds, for the start of the clip in the media.
CLIP_START_TIMESTAMP_KEY = "clip/start/timestamp"
# The end time, in microseconds, for the end of the clip in the media.
CLIP_END_TIMESTAMP_KEY = "clip/end/timestamp"
# A list of label indices for this clip.
CLIP_LABEL_INDEX_KEY = "clip/label/index"
# A list of label strings for this clip.
CLIP_LABEL_STRING_KEY = "clip/label/string"
# A list of label confidences for this clip.
CLIP_LABEL_CONFIDENCE_KEY = "clip/label/confidence"
# A list of label start timestamps for this clip.
CLIP_LABEL_START_TIMESTAMP_KEY = "clip/label/start/timestamp"
# A list of label end timestamps for this clip.
CLIP_LABEL_END_TIMESTAMP_KEY = "clip/label/end/timestamp"
msu.create_bytes_context_feature(
    "example_id", EXAMPLE_ID_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "example_dataset_name", EXAMPLE_DATASET_NAME_KEY, module_dict=globals())
msu.create_bytes_list_context_feature(
    "example_dataset_flag_string", EXAMPLE_DATASET_FLAG_STRING_KEY,
    module_dict=globals())
msu.create_bytes_context_feature(
    "clip_media_id", CLIP_MEDIA_ID_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "clip_alternative_media_id", ALTERNATIVE_CLIP_MEDIA_ID_KEY,
    module_dict=globals())
msu.create_bytes_context_feature(
    "clip_encoded_media_bytes", CLIP_ENCODED_MEDIA_BYTES_KEY,
    module_dict=globals())
msu.create_bytes_context_feature(
    "clip_data_path", CLIP_DATA_PATH_KEY, module_dict=globals())
msu.create_int_context_feature(
    "clip_encoded_media_start_timestamp",
    CLIP_ENCODED_MEDIA_START_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_context_feature(
    "clip_start_timestamp", CLIP_START_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_context_feature(
    "clip_end_timestamp", CLIP_END_TIMESTAMP_KEY, module_dict=globals())
msu.create_bytes_list_context_feature(
    "clip_label_string", CLIP_LABEL_STRING_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "clip_label_index", CLIP_LABEL_INDEX_KEY, module_dict=globals())
msu.create_float_list_context_feature(
    "clip_label_confidence", CLIP_LABEL_CONFIDENCE_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "clip_label_start_timestamp",
    CLIP_LABEL_START_TIMESTAMP_KEY,
    module_dict=globals())
msu.create_int_list_context_feature(
    "clip_label_end_timestamp",
    CLIP_LABEL_END_TIMESTAMP_KEY,
    module_dict=globals())

##################################  SEGMENTS  #################################
# A list of segment start times in microseconds.
SEGMENT_START_TIMESTAMP_KEY = "segment/start/timestamp"
# A list of indices marking the first frame index >= the start timestamp.
SEGMENT_START_INDEX_KEY = "segment/start/index"
# A list of segment end times in microseconds.
SEGMENT_END_TIMESTAMP_KEY = "segment/end/timestamp"
# A list of indices marking the last frame index <= the end timestamp.
SEGMENT_END_INDEX_KEY = "segment/end/index"
# A list with the label index for each segment.
# Multiple labels for the same segment are encoded as repeated segments.
SEGMENT_LABEL_INDEX_KEY = "segment/label/index"
# A list with the label string for each segment.
# Multiple labels for the same segment are encoded as repeated segments.
SEGMENT_LABEL_STRING_KEY = "segment/label/string"
# A list with the label confidence for each segment.
# Multiple labels for the same segment are encoded as repeated segments.
SEGMENT_LABEL_CONFIDENCE_KEY = "segment/label/confidence"
msu.create_bytes_list_context_feature(
    "segment_label_string", SEGMENT_LABEL_STRING_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_start_timestamp",
    SEGMENT_START_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_start_index", SEGMENT_START_INDEX_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_end_timestamp", SEGMENT_END_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_end_index", SEGMENT_END_INDEX_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_label_index", SEGMENT_LABEL_INDEX_KEY, module_dict=globals())
msu.create_float_list_context_feature(
    "segment_label_confidence",
    SEGMENT_LABEL_CONFIDENCE_KEY, module_dict=globals())

##########################  REGIONS / BOUNDING BOXES  #########################

# Normalized coordinates of bounding boxes are provided in four lists to avoid
# order ambiguity. We provide additional accessors for complete bounding boxes
# below.
REGION_BBOX_YMIN_KEY = "region/bbox/ymin"
REGION_BBOX_XMIN_KEY = "region/bbox/xmin"
REGION_BBOX_YMAX_KEY = "region/bbox/ymax"
REGION_BBOX_XMAX_KEY = "region/bbox/xmax"
# The point and radius can denote keypoints.
REGION_POINT_X_KEY = "region/point/x"
REGION_POINT_Y_KEY = "region/point/y"
REGION_RADIUS_KEY = "region/radius"
# The 3D point can denote keypoints.
REGION_3D_POINT_X_KEY = "region/3d_point/x"
REGION_3D_POINT_Y_KEY = "region/3d_point/y"
REGION_3D_POINT_Z_KEY = "region/3d_point/z"
# The number of regions at that timestep.
REGION_NUM_REGIONS_KEY = "region/num_regions"
# Whether that timestep is annotated for regions.
# (Disambiguates between multiple meanings of num_regions = 0.)
REGION_IS_ANNOTATED_KEY = "region/is_annotated"
# A list indicating if each region is generated (1) or manually annotated (0)
REGION_IS_GENERATED_KEY = "region/is_generated"
# A list indicating if each region is occluded (1) or visible (0)
REGION_IS_OCCLUDED_KEY = "region/is_occluded"
# Lists with a label for each region.
# Multiple labels for the same region require duplicating the region.
REGION_LABEL_INDEX_KEY = "region/label/index"
REGION_LABEL_STRING_KEY = "region/label/string"
REGION_LABEL_CONFIDENCE_KEY = "region/label/confidence"
# Lists with a track identifier for each region.
# Multiple track identifier for the same region require duplicating the region.
REGION_TRACK_INDEX_KEY = "region/track/index"
REGION_TRACK_STRING_KEY = "region/track/string"
REGION_TRACK_CONFIDENCE_KEY = "region/track/confidence"
# Lists with a class for each region. In general, prefer to use the label
# fields. These class fields exist to distinguish tracks when different classes
# have overlapping track ids.
REGION_CLASS_INDEX_KEY = "region/class/index"
REGION_CLASS_STRING_KEY = "region/class/string"
REGION_CLASS_CONFIDENCE_KEY = "region/class/confidence"
# The timestamp of the region annotation in microseconds.
REGION_TIMESTAMP_KEY = "region/timestamp"
# The original timestamp in microseconds for region annotations.
# If regions are aligned to image frames, this field preserves the original
# timestamps.
REGION_UNMODIFIED_TIMESTAMP_KEY = "region/unmodified_timestamp"
# The list of region parts expected in this example.
REGION_PARTS_KEY = "region/parts"
# The dimensions of each embedding per region / bounding box.
REGION_EMBEDDING_DIMENSIONS_PER_REGION_KEY = (
    "region/embedding/dimensions_per_region")
# The format encoding embeddings as strings.
REGION_EMBEDDING_FORMAT_KEY = "region/embedding/format"
# An embedding for each region. The length of each list must be the product of
# the number of regions and the product of the embedding dimensions.
REGION_EMBEDDING_FLOAT_KEY = "region/embedding/float"
# A string encoded embedding for each regions.
REGION_EMBEDDING_ENCODED_KEY = "region/embedding/encoded"
# The confidence of the embedding.
REGION_EMBEDDING_CONFIDENCE_KEY = "region/embedding/confidence"


def _create_region_with_prefix(name, prefix):
  """Create multiple accessors for region based data."""
  msu.create_int_feature_list(name + "_num_regions", REGION_NUM_REGIONS_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(name + "_is_annotated", REGION_IS_ANNOTATED_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_is_occluded", REGION_IS_OCCLUDED_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_is_generated", REGION_IS_GENERATED_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(name + "_timestamp", REGION_TIMESTAMP_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(
      name + "_unmodified_timestamp", REGION_UNMODIFIED_TIMESTAMP_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(
      name + "_label_string", REGION_LABEL_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_label_index", REGION_LABEL_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_label_confidence", REGION_LABEL_CONFIDENCE_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(
      name + "_class_string", REGION_CLASS_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_class_index", REGION_CLASS_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_class_confidence", REGION_CLASS_CONFIDENCE_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(
      name + "_track_string", REGION_TRACK_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_track_index", REGION_TRACK_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_track_confidence", REGION_TRACK_CONFIDENCE_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_ymin", REGION_BBOX_YMIN_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_xmin", REGION_BBOX_XMIN_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_ymax", REGION_BBOX_YMAX_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_xmax", REGION_BBOX_XMAX_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_point_x", REGION_POINT_X_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_point_y", REGION_POINT_Y_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_3d_point_x", REGION_3D_POINT_X_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_3d_point_y", REGION_3D_POINT_Y_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_3d_point_z", REGION_3D_POINT_Z_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_context_feature(name + "_parts",
                                        REGION_PARTS_KEY,
                                        prefix=prefix, module_dict=globals())
  msu.create_float_list_context_feature(
      name + "_embedding_dimensions_per_region",
      REGION_EMBEDDING_DIMENSIONS_PER_REGION_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_context_feature(name + "_embedding_format",
                                   REGION_EMBEDDING_FORMAT_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_embedding_floats",
                                     REGION_EMBEDDING_FLOAT_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(name + "_embedding_encoded",
                                     REGION_EMBEDDING_ENCODED_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_embedding_confidence",
                                     REGION_EMBEDDING_CONFIDENCE_KEY,
                                     prefix=prefix, module_dict=globals())
  # pylint: disable=undefined-variable
  def get_prefixed_bbox_at(index, sequence_example, prefix):
    return np.stack((
        get_bbox_ymin_at(index, sequence_example, prefix=prefix),
        get_bbox_xmin_at(index, sequence_example, prefix=prefix),
        get_bbox_ymax_at(index, sequence_example, prefix=prefix),
        get_bbox_xmax_at(index, sequence_example, prefix=prefix)),
                    1)
  def add_prefixed_bbox(values, sequence_example, prefix):
    values = np.array(values)
    if values.size == 0:
      add_bbox_ymin([], sequence_example, prefix=prefix)
      add_bbox_xmin([], sequence_example, prefix=prefix)
      add_bbox_ymax([], sequence_example, prefix=prefix)
      add_bbox_xmax([], sequence_example, prefix=prefix)
    else:
      add_bbox_ymin(values[:, 0], sequence_example, prefix=prefix)
      add_bbox_xmin(values[:, 1], sequence_example, prefix=prefix)
      add_bbox_ymax(values[:, 2], sequence_example, prefix=prefix)
      add_bbox_xmax(values[:, 3], sequence_example, prefix=prefix)
  def get_prefixed_bbox_size(sequence_example, prefix):
    return get_bbox_ymin_size(sequence_example, prefix=prefix)
  def has_prefixed_bbox(sequence_example, prefix):
    return has_bbox_ymin(sequence_example, prefix=prefix)
  def clear_prefixed_bbox(sequence_example, prefix):
    clear_bbox_ymin(sequence_example, prefix=prefix)
    clear_bbox_xmin(sequence_example, prefix=prefix)
    clear_bbox_ymax(sequence_example, prefix=prefix)
    clear_bbox_xmax(sequence_example, prefix=prefix)
  def get_prefixed_point_at(index, sequence_example, prefix):
    return np.stack((
        get_bbox_point_y_at(index, sequence_example, prefix=prefix),
        get_bbox_point_x_at(index, sequence_example, prefix=prefix)),
                    1)
  def add_prefixed_point(values, sequence_example, prefix):
    add_bbox_point_y(values[:, 0], sequence_example, prefix=prefix)
    add_bbox_point_x(values[:, 1], sequence_example, prefix=prefix)
  def get_prefixed_point_size(sequence_example, prefix):
    return get_bbox_point_y_size(sequence_example, prefix=prefix)
  def has_prefixed_point(sequence_example, prefix):
    return has_bbox_point_y(sequence_example, prefix=prefix)
  def clear_prefixed_point(sequence_example, prefix):
    clear_bbox_point_y(sequence_example, prefix=prefix)
    clear_bbox_point_x(sequence_example, prefix=prefix)
  def get_prefixed_3d_point_at(index, sequence_example, prefix):
    return np.stack((
        get_bbox_3d_point_x_at(index, sequence_example, prefix=prefix),
        get_bbox_3d_point_y_at(index, sequence_example, prefix=prefix),
        get_bbox_3d_point_z_at(index, sequence_example, prefix=prefix)),
                    1)
  def add_prefixed_3d_point(values, sequence_example, prefix):
    add_bbox_3d_point_x(values[:, 0], sequence_example, prefix=prefix)
    add_bbox_3d_point_y(values[:, 1], sequence_example, prefix=prefix)
    add_bbox_3d_point_z(values[:, 2], sequence_example, prefix=prefix)
  def get_prefixed_3d_point_size(sequence_example, prefix):
    return get_bbox_3d_point_x_size(sequence_example, prefix=prefix)
  def has_prefixed_3d_point(sequence_example, prefix):
    return has_bbox_3d_point_x(sequence_example, prefix=prefix)
  def clear_prefixed_3d_point(sequence_example, prefix):
    clear_bbox_3d_point_x(sequence_example, prefix=prefix)
    clear_bbox_3d_point_y(sequence_example, prefix=prefix)
    clear_bbox_3d_point_z(sequence_example, prefix=prefix)
  # pylint: enable=undefined-variable
  msu.add_functions_to_module({
      "get_" + name + "_at":
          msu.function_with_default(get_prefixed_bbox_at, prefix),
      "add_" + name:
          msu.function_with_default(add_prefixed_bbox, prefix),
      "get_" + name + "_size":
          msu.function_with_default(get_prefixed_bbox_size, prefix),
      "has_" + name:
          msu.function_with_default(has_prefixed_bbox, prefix),
      "clear_" + name:
          msu.function_with_default(clear_prefixed_bbox, prefix),
  }, module_dict=globals())
  msu.add_functions_to_module({
      "get_" + name + "_point_at":
          msu.function_with_default(get_prefixed_point_at, prefix),
      "add_" + name + "_point":
          msu.function_with_default(add_prefixed_point, prefix),
      "get_" + name + "_point_size":
          msu.function_with_default(get_prefixed_point_size, prefix),
      "has_" + name + "_point":
          msu.function_with_default(has_prefixed_point, prefix),
      "clear_" + name + "_point":
          msu.function_with_default(clear_prefixed_point, prefix),
  }, module_dict=globals())
  msu.add_functions_to_module({
      "get_" + name + "_3d_point_at":
          msu.function_with_default(get_prefixed_3d_point_at, prefix),
      "add_" + name + "_3d_point":
          msu.function_with_default(add_prefixed_3d_point, prefix),
      "get_" + name + "_3d_point_size":
          msu.function_with_default(get_prefixed_3d_point_size, prefix),
      "has_" + name + "_3d_point":
          msu.function_with_default(has_prefixed_3d_point, prefix),
      "clear_" + name + "_3d_point":
          msu.function_with_default(clear_prefixed_3d_point, prefix),
  }, module_dict=globals())


PREDICTED_PREFIX = "PREDICTED"
_create_region_with_prefix("bbox", "")
_create_region_with_prefix("predicted_bbox", PREDICTED_PREFIX)


###################################  IMAGES  #################################
# The format the images are encoded as (e.g. "JPEG", "PNG")
IMAGE_FORMAT_KEY = "image/format"
# The number of channels in the image.
IMAGE_CHANNELS_KEY = "image/channels"
# The colorspace of the iamge.
IMAGE_COLORSPACE_KEY = "image/colorspace"
# The height of the image in pixels.
IMAGE_HEIGHT_KEY = "image/height"
# The width of the image in pixels.
IMAGE_WIDTH_KEY = "image/width"
# frame rate in images/second of media.
IMAGE_FRAME_RATE_KEY = "image/frame_rate"
# The maximum values if the images were saturated and normalized for encoding.
IMAGE_SATURATION_KEY = "image/saturation"
# The listing from discrete image values (as indices) to class indices.
IMAGE_CLASS_LABEL_INDEX_KEY = "image/class/label/index"
# The listing from discrete image values (as indices) to class strings.
IMAGE_CLASS_LABEL_STRING_KEY = "image/class/label/string"
# The listing from discrete instance indices to class indices they embody.
IMAGE_OBJECT_CLASS_INDEX_KEY = "image/object/class/index"
# The encoded image frame.
IMAGE_ENCODED_KEY = "image/encoded"
# Multiple images from the same timestep (e.g. multiview video).
IMAGE_MULTI_ENCODED_KEY = "image/multi_encoded"
# The timestamp of the frame in microseconds.
IMAGE_TIMESTAMP_KEY = "image/timestamp"
# A per image label if specific frames have labels.
# If time spans have labels, segments are preferred to allow changing rates.
IMAGE_LABEL_INDEX_KEY = "image/label/index"
IMAGE_LABEL_STRING_KEY = "image/label/string"
IMAGE_LABEL_CONFIDENCE_KEY = "image/label/confidence"
# The path of the image file if it did not come from a media clip.
IMAGE_DATA_PATH_KEY = "image/data_path"


def _create_image_with_prefix(name, prefix):
  """Create multiple accessors for image based data."""
  msu.create_bytes_context_feature(name + "_format", IMAGE_FORMAT_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_bytes_context_feature(name + "_colorspace", IMAGE_COLORSPACE_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_int_context_feature(name + "_channels", IMAGE_CHANNELS_KEY,
                                 prefix=prefix, module_dict=globals())
  msu.create_int_context_feature(name + "_height", IMAGE_HEIGHT_KEY,
                                 prefix=prefix, module_dict=globals())
  msu.create_int_context_feature(name + "_width", IMAGE_WIDTH_KEY,
                                 prefix=prefix, module_dict=globals())
  msu.create_bytes_feature_list(name + "_encoded", IMAGE_ENCODED_KEY,
                                prefix=prefix, module_dict=globals())
  msu.create_float_context_feature(name + "_frame_rate", IMAGE_FRAME_RATE_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_bytes_list_context_feature(
      name + "_class_label_string", IMAGE_CLASS_LABEL_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_context_feature(
      name + "_class_label_index", IMAGE_CLASS_LABEL_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_context_feature(
      name + "_object_class_index", IMAGE_OBJECT_CLASS_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_context_feature(name + "_data_path", IMAGE_DATA_PATH_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(name + "_timestamp", IMAGE_TIMESTAMP_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(name + "_multi_encoded",
                                     IMAGE_MULTI_ENCODED_KEY, prefix=prefix,
                                     module_dict=globals())
FORWARD_FLOW_PREFIX = "FORWARD_FLOW"
CLASS_SEGMENTATION_PREFIX = "CLASS_SEGMENTATION"
INSTANCE_SEGMENTATION_PREFIX = "INSTANCE_SEGMENTATION"
_create_image_with_prefix("image", "")
_create_image_with_prefix("forward_flow", FORWARD_FLOW_PREFIX)
_create_image_with_prefix("class_segmentation", CLASS_SEGMENTATION_PREFIX)
_create_image_with_prefix("instance_segmentation", INSTANCE_SEGMENTATION_PREFIX)

##################################  TEXT  #################################
# Which language text tokens are likely to be in.
TEXT_LANGUAGE_KEY = "text/language"
# A large block of text that applies to the media.
TEXT_CONTEXT_CONTENT_KEY = "text/context/content"
# A large block of text that applies to the media as token ids.
TEXT_CONTEXT_TOKEN_ID_KEY = "text/context/token_id"
# A large block of text that applies to the media as embeddings.
TEXT_CONTEXT_EMBEDDING_KEY = "text/context/embedding"

# The text contents for a given time.
TEXT_CONTENT_KEY = "text/content"
# The start time for the text becoming relevant.
TEXT_TIMESTAMP_KEY = "text/timestamp"
# The duration where the text is relevant.
TEXT_DURATION_KEY = "text/duration"
# The confidence that this is the correct text.
TEXT_CONFIDENCE_KEY = "text/confidence"
# A floating point embedding corresponding to the text.
TEXT_EMBEDDING_KEY = "text/embedding"
# An integer id corresponding to the text.
TEXT_TOKEN_ID_KEY = "text/token/id"

msu.create_bytes_context_feature(
    "text_language", TEXT_LANGUAGE_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "text_context_content", TEXT_CONTEXT_CONTENT_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "text_context_token_id", TEXT_CONTEXT_TOKEN_ID_KEY, module_dict=globals())
msu.create_float_list_context_feature(
    "text_context_embedding", TEXT_CONTEXT_EMBEDDING_KEY, module_dict=globals())
msu.create_bytes_feature_list(
    "text_content", TEXT_CONTENT_KEY, module_dict=globals())
msu.create_int_feature_list(
    "text_timestamp", TEXT_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_feature_list(
    "text_duration", TEXT_DURATION_KEY, module_dict=globals())
msu.create_float_feature_list(
    "text_confidence", TEXT_CONFIDENCE_KEY, module_dict=globals())
msu.create_float_list_feature_list(
    "text_embedding", TEXT_EMBEDDING_KEY, module_dict=globals())
msu.create_int_feature_list(
    "text_token_id", TEXT_TOKEN_ID_KEY, module_dict=globals())

##################################  FEATURES  #################################
# The dimensions of the feature.
FEATURE_DIMENSIONS_KEY = "feature/dimensions"
# The rate the features are extracted per second of media.
FEATURE_RATE_KEY = "feature/rate"
# The encoding format if any for the feature.
FEATURE_BYTES_FORMAT_KEY = "feature/bytes/format"
# For audio, the rate the samples are extracted per second of media.
FEATURE_SAMPLE_RATE_KEY = "feature/sample_rate"
# For audio, the number of channels per extracted feature.
FEATURE_NUM_CHANNELS_KEY = "feature/num_channels"
# For audio, th enumber of samples per extracted feature.
FEATURE_NUM_SAMPLES_KEY = "feature/num_samples"
# For audio, the rate the features are extracted per second of media.
FEATURE_PACKET_RATE_KEY = "feature/packet_rate"
# For audio, the original audio sampling rate the feature is derived from.
FEATURE_AUDIO_SAMPLE_RATE_KEY = "feature/audio_sample_rate"
# The feature as a list of floats.
FEATURE_FLOATS_KEY = "feature/floats"
# The feature as a list of bytes. May be encoded.
FEATURE_BYTES_KEY = "feature/bytes"
# The feature as a list of ints.
FEATURE_INTS_KEY = "feature/ints"
# The timestamp, in microseconds, of the feature.
FEATURE_TIMESTAMP_KEY = "feature/timestamp"
# It is occasionally useful to indicate that a feature applies to a given range.
# This should be used for features only and annotations should be provided as
# segments.
FEATURE_DURATION_KEY = "feature/duration"
# Encodes an optional confidence score for the generated features.
FEATURE_CONFIDENCE_KEY = "feature/confidence"
# The feature as a list of floats in the context.
CONTEXT_FEATURE_FLOATS_KEY = "context_feature/floats"
# The feature as a list of bytes in the context. May be encoded.
CONTEXT_FEATURE_BYTES_KEY = "context_feature/bytes"
# The feature as a list of ints in the context.
CONTEXT_FEATURE_INTS_KEY = "context_feature/ints"

msu.create_int_list_context_feature(
    "feature_dimensions", FEATURE_DIMENSIONS_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_rate", FEATURE_RATE_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "feature_bytes_format", FEATURE_BYTES_FORMAT_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_sample_rate", FEATURE_SAMPLE_RATE_KEY, module_dict=globals())
msu.create_int_context_feature(
    "feature_num_channels", FEATURE_NUM_CHANNELS_KEY, module_dict=globals())
msu.create_int_context_feature(
    "feature_num_samples", FEATURE_NUM_SAMPLES_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_packet_rate", FEATURE_PACKET_RATE_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_audio_sample_rate", FEATURE_AUDIO_SAMPLE_RATE_KEY,
    module_dict=globals())
msu.create_float_list_feature_list(
    "feature_floats", FEATURE_FLOATS_KEY, module_dict=globals())
msu.create_bytes_list_feature_list(
    "feature_bytes", FEATURE_BYTES_KEY, module_dict=globals())
msu.create_int_list_feature_list(
    "feature_ints", FEATURE_INTS_KEY, module_dict=globals())
msu.create_int_feature_list(
    "feature_timestamp", FEATURE_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_list_feature_list(
    "feature_duration", FEATURE_DURATION_KEY, module_dict=globals())
msu.create_float_list_feature_list(
    "feature_confidence", FEATURE_CONFIDENCE_KEY, module_dict=globals())
msu.create_float_list_context_feature(
    "context_feature_floats", CONTEXT_FEATURE_FLOATS_KEY, module_dict=globals())
msu.create_bytes_list_context_feature(
    "context_feature_bytes", CONTEXT_FEATURE_BYTES_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "context_feature_ints", CONTEXT_FEATURE_INTS_KEY, module_dict=globals())

