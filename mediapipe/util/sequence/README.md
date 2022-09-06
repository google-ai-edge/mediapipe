# MediaSequence keys and functions reference

The documentation below will first provide an overview of using MediaSequence
for machine learning tasks. Then, the documentation will describe the function
prototypes used in MediaSequence for storing multimedia data in
SequenceExamples. Finally, the documentation will describe the specific keys for
storing specific types of data.

## Overview of MediaSequence for machine learning

The goal of MediaSequence is to provide a tool for transforming annotations of
multimedia into input examples ready for use with machine learning models in
TensorFlow. The most semantically appropriate data type for this task that can
be easily parsed in TensorFlow is
tensorflow.train.SequenceExamples/tensorflow::SequenceExamples.
Using SequenceExamples enables quick integration of new
features into TensorFlow pipelines, easy open sourcing of models and data,
reasonable debugging, and efficient TensorFlow decoding. For many machine
learning tasks, TensorFlow Examples are capable of fulfilling that role.
However, Examples can become unwieldy for sequence data, particularly when the
number of features per timestep varies, creating a ragged struction. Video
object detection is one example task that requires this ragged structure because
the number of detections per frame varies. SequenceExamples can easily encode
this ragged structure. Sequences naturally match the semantics of video as a
sequence of frames or other common media patterns. The video feature lists will
be stored in order with strictly increasing timestamps so the data is
unambiguously ordered. The interpretable semantics simplify debugging and
decoding of potentially complicated data. One potential disadvantage of
SequenceExamples is that keys and formats can vary widely. The MediaSequence
library provides tools for consistently manipulating and decoding
SequenceExamples in Python and C++ in a consistent format. The consistent format
enables creating a pipeline for processing data sets. A goal of MediaSequence as
a pipeline is that users should only need to specify the metadata (e.g. videos
and labels) for their task. The pipeline will turn the metadata into training
data.

The pipeline has two stages. First, users must generate the metadata
describing the data and applicable labels. This process is
straightforward and described in the next section. Second, users run MediaPipe
graphs with the `UnpackMediaSequenceCalculator` and
`PackMediaSequenceCalculators` to extract the relevant data from multimedia
files. A sequence of graphs can be chained together in this second stage to
achieve complex processing such as first extracting a subset of frames from a
video and then extracting deep features or object detections for each extracted
frame. As MediaPipe is built to simply and reproducibly process media files,
the two stage approach separates and simplifies data management.

### Creating metadata for a new data set

Generating examples for a new data set typically only requires defining the
metadata. MediaPipe graphs can interpret this metadata to fill out the
SequenceExamples using the `UnpackMediaSequenceCalculator` and
`PackMediaSequenceCalculator`. This section will list the metadata required for
different types of tasks and provide a limited descripiton for the data filled
by MediaPipe. The input media will be referred to as video because that is a
common case, but audio files or other sequences could be supported. The function
calls in the Python API will be used in examples, and the equivalent C++ calls
are described below.

The video metadata is a way to access the video, using `set_clip_data_path` to
define the path on disk, and the time span to include using
`set_clip_start_timestamp` and `set_clip_end_timestamp`. The data path can be
absolute or can be relative to a root directory passed to the
`UnpackMediaSequenceCalculator`. The start and end timestamps should be valid
MediaPipe timestamps in microseconds. Given this information, the pipeline can
extract the portion of the media between the start and end timestamps. If you do
not specify a start time, the video is decoded from the beginning. If you do not
specify an end time, the entire video is decoded. The start and end times are
not filled if left empty.

The features extracted from the video depends on the MediaPipe graph that is
run. The documentation of keys below and in `PackMediaSequenceCalculator`
provide the best description.

The annotations including labels should be added as metadata. They will be
passed through the MediaPipe pipeline unchanged. The label format will vary
depending on the task you want to do. Several examples are included below. In
general, the MediaPipe processing is independent of any labels that you provide:
only the clip data path, start time, and end time matter.

#### Clip classification

For clip classification, e.g. is this video clip about basketball?, you
should use `set_clip_label_index` with the integer index of the correct class
and `set_clip_label_string` with the human readable version of the correct class.
The index is often used when training the model and the string is used for
human readable debugging. The same number of indices and strings need to be
provided. The association between the two is just their relative positions in
the list.

##### Example lines creating metadata for clip classification

```python
# Python: functions from media_sequence.py as ms
sequence = tf.train.SequenceExample()
ms.set_clip_data_path(b"path_to_video", sequence)
ms.set_clip_start_timestamp(1000000, sequence)
ms.set_clip_end_timestamp(6000000, sequence)
ms.set_clip_label_index((4, 3), sequence)
ms.set_clip_label_string((b"run", b"jump"), sequence)
```

```c++
// C++: functions from media_sequence.h
tensorflow::SequenceExample sequence;
SetClipDataPath("path_to_video", &sequence);
SetClipStartTimestamp(1000000, &sequence);
SetClipEndTimestamp(6000000, &sequence);
SetClipLabelIndex({4, 3}, &sequence);
SetClipLabelString({"run", "jump"}, &sequence);
```

#### Temporal detection

For temporal event detection or localization, e.g. classify regions in time
where people are playing a sport, the labels are referred to as segments. You
need to set the segment timespans with `set_segment_start_timestamp` and
`set_segment_end_timestamp` and labels with `set_segment_label_index` and
`set_segment_label_string`. All of these are repeated fields so you can provide
multiple segments for each clip. The label index and string have the same
meaning as for clip classification. Only the start and end timestamps need to
be provided. (The pipeline will automatically call `set_segment_start_index` to
the index of the image frame under the image/timestamp key that is closest in
time, and similarly for `set_segment_end_index`. Allowing the pipeline to fill
in the indices corrects for frame rate changes automatically.) The same number
of values must be present in each field. If the same segment would have
multiple labels, the segment start and end time must be duplicated.

##### Example lines creating metadata for temporal detection
```python
# Python: functions from media_sequence.py as ms
sequence = tf.train.SequenceExample()
ms.set_clip_data_path(b"path_to_video", sequence)
ms.set_clip_start_timestamp(1000000, sequence)
ms.set_clip_end_timestamp(6000000, sequence)

ms.set_segment_start_timestamp((2000000, 4000000), sequence)
ms.set_segment_end_timestamp((3500000, 6000000), sequence)
ms.set_segment_label_index((4, 3), sequence)
ms.set_segment_label_string((b"run", b"jump"), sequence)
```

```c++
// C++: functions from media_sequence.h
tensorflow::SequenceExample sequence;
SetClipDataPath("path_to_video", &sequence);
SetClipStartTimestamp(1000000, &sequence);
SetClipEndTimestamp(6000000, &sequence);

SetSegmentStartTimestamp({2000000, 4000000}, &sequence);
SetSegmentEndTimestamp({3500000, 6000000}, &sequence);
SetSegmentLabelIndex({4, 3}, &sequence);
SetSegmentLabelString({"run", "jump"}, &sequence);
```

#### Tracking and spatiotemporal detection

For object tracking or detection in videos, e.g. classify regions in time and
space, the labels are typically bounding boxes. Unlike previous tasks, the
annotations are provided as a [`FeatureList`](https://www.tensorflow.org/api_docs/python/tf/train/FeatureList)
instead of in a context [`Feature`](https://www.tensorflow.org/api_docs/python/tf/train/Feature)
because they occur in multiple frames. Set up a detection task with `add_bbox`,
`add_bbox_timestamp`, `add_bbox_label_string`, and `add_bbox_label_index`. Only
add metadata for annotated frames. The pipeline will add empty features to each
feature list to align the box annotations with the nearest image frame.
`add_bbox_is_annotated` distinguishes between annotated frames and frames added
as padding. 1 is added if the frame was annotated and 0 otherwise. It is
automatically maintained in `PackMediaSequenceCalculator`. Other fields can be
used for tracking tasks: `add_bbox_track_string` identifies instances over time
and `add_bbox_class_string` can be concatenated to the track string if track ids
are not already unique. If track ids are unique across classes, you do not need
to fill out the class information.

##### Example lines creating metadata for spatiotemporal detection or tracking

```python
# Python: functions from media_sequence.py as ms
sequence = tf.train.SequenceExample()
ms.set_clip_data_path(b"path_to_video", sequence)
ms.set_clip_start_timestamp(1000000, sequence)
ms.set_clip_end_timestamp(6000000, sequence)

# For an object tracking task with action labels:
loctions_on_frame_1 = np.array([[0.1, 0.2, 0.3 0.4],
                                [0.2, 0.3, 0.4, 0.5]])
ms.add_bbox(locations_on_frame_1, sequence)
ms.add_bbox_timestamp(3000000, sequence)
ms.add_bbox_label_index((4, 3), sequence)
ms.add_bbox_label_string((b"run", b"jump"), sequence)
ms.add_bbox_track_string((b"id_0", b"id_1"), sequence)
# ms.add_bbox_class_string(("cls_0", "cls_0"), sequence)  # if required
locations_on_frame_2 = locations_on_frame_1[0]
ms.add_bbox(locations_on_frame_2, sequence)
ms.add_bbox_timestamp(5000000, sequence)
ms.add_bbox_label_index((3), sequence)
ms.add_bbox_label_string((b"jump",), sequence)
ms.add_bbox_track_string((b"id_0",), sequence)
# ms.add_bbox_class_string(("cls_0",), sequence)  # if required
```

```c++
// C++: functions from media_sequence.h
tensorflow::SequenceExample sequence;
SetClipDataPath("path_to_video", &sequence);
SetClipStartTimestamp(1000000, &sequence);
SetClipEndTimestamp(6000000, &sequence);

// For an object tracking task with action labels:
std::vector<mediapipe::Location> locations_on_frame_1;
AddBBox(locations_on_frame_1, &sequence);
AddBBoxTimestamp(3000000, &sequence);
AddBBoxLabelIndex({4, 3}, &sequence);
AddBBoxLabelString({"run", "jump"}, &sequence);
AddBBoxTrackString({"id_0", "id_1"}, &sequence);
// AddBBoxClassString({"cls_0", "cls_0"}, &sequence); // if required
std::vector<mediapipe::Location> locations_on_frame_2;
AddBBox(locations_on_frame_2, &sequence);
AddBBoxTimestamp(5000000, &sequence);
AddBBoxLabelIndex({3}, &sequence);
AddBBoxLabelString({"jump"}, &sequence);
AddBBoxTrackString({"id_0"}, &sequence);
// AddBBoxClassString({"cls_0"}, &sequence); // if required
```

### Running a MediaSequence through MediaPipe

#### `UnpackMediaSequenceCalculator` and `PackMediaSequenceCalculator`
MediaSequence utilizes MediaPipe for processing by providing two special
calculators. The `UnpackMediaSequenceCalculator` is used to extract data from
SequenceExamples. This will often be the metadata, such as the path to the video
file, and the clip start and end times. However, after storing images in a
SequenceExample, the images themselves can also be unpacked for further
processing, such as computing optical flow. Whatever data is extracted during
processing is added to the SequenceExample by the `PackMediaSequenceCalculator`.
The values that are unpacked and packed into these calculators are determined
by the tags on the streams in the MediaPipe calculator graph. (Tags are required
to be all capitals and underscores. To encode prefixes for feature keys as tags,
prefixes for feature keys should follow the same convention.) The documentation
for these two calculators describes the variety of data they support. The
timestamps of each feature list being unpacked must be in strictly increasing
order. Any other MediaPipe processing can be used between these calculators to
extract features.

#### Adding data and reconciling metadata
In general, the pipeline will decode the specified media between the clip
start and end timestamps and store any requested features. A common feature
to request is JPEG encoded images, so this will be used it as an example.
Each image between the clip start and end timestamps (generally inclusive) is
added to the SequenceExample's feature list with add_image_encoded and the
corresponding timestamp it arrived at is added with add_image_timestamp. At the
end of the image stream, the pipeline will determine and store what metadata it
can about the stream. For images, it will store the height and width of the
image as well as the number of channels and encoding format. Similar storage and
metadata computation is done when adding audio, float feature vectors, or
encoded optical flow to the SequenceExample. The code that reconciles the
metadata is in media_sequence.cc.

#### Automatically aligning bounding boxes to images
At the time of writing, the image/timestamp is also used to update the closest
timestamp for segment/start/index and segment/end/index and bounding box data.
Segment indices are relative to the start of the clip (i.e. only reference data
within the SequenceExample), while timestamps are absolute times within
the video. Bounding box data is aligned to the image/timestamps by inserting
empty bounding box annotations and indicating this with add_bbox_is_annotated.
If images are stored at a lower rate than the bounding box data, then only the
nearest annotation to each frame is retained and any others are dropped. *Be
careful when downsampling frame rates with bounding box annotations;
downsampling bounding box annotations is the only time annotations will be lost
in the MediaPipe pipeline.*

#### Chaining processing graphs
A common use case is to derive deep features from frames in a video when those
features are too expensive to compute during training. For example, extracting
ResNet-50 features on each frame of video. In the MediaSequence pipeline, the
way to generate these features is to first extract the images to the
SequenceExample in one MediaPipe graph. Then create a second MediaPipe graph
that unpacks the images from the SequenceExample and appends the new features to
a copy of that SequenceExample. This chaining behavior makes it easy to
incrementally add features in a modular way and makes debugging easier because
you can identify the anomalous stage more easily. Once the pipeline is complete,
any unnecessary features can be removed. Be aware that the number of derived
feature timestamps may be different than the number of input features, e.g.
optical flow can't be estimated for the last frame of a video clip, so it
adds one less frame of data. With the exception of aligning bounding boxes, the
pipeline does nothing to require consistent timestamps between features.

### Using prefixes

Prefixes enable storing semantically identical data without collisions. For
example, it is possible to store predicted and ground truth bounding boxes by
using different prefixes. We can also store bounding boxes and labels from
different tasks by utilizing prefixes.

To minimize burdening the API and documentation, eschew using prefixes unless
necessary.

The recommended prefix format, enforced by some MediaPipe functions, is all caps
with underscores, and numeric characters after the first character. e.g.
`MY_FAVORITE_FEATURE_V1`.

The convention for encoding groundtruth labels is to use no prefix, while
predicted labels are typically tagged with prefixes. For example:

*   Example groudntruth keys:
    *   `region/label/string`
    *   `region/label/confidence`

*   Example predicted label keys:
    *   `PREDICT_V1/region/label/string`
    *   `PREDICT_V1/region/label/confidence`

## Function prototypes for each data type

MediaSequence provides accessors to store common data patterns in
SequenceExamples. The exact functions depend on the type of data
and the key, but the patterns are similar. Each function has a name related to
the key, so we will document the functions with a generic name, Feature. Note
that due to different conventions for Python and C++ code, the capitalization
and parameter order varies, but the functionality should be equivalent.

Each function takes an optional prefix parameter. For some common cases, such as
storing instance segmentation labels along with images, named versions with
prefixes baked in provided as documented below. Lastly, generic features and
audio streams should almost always use a prefix because storing multiple
features or transformed audio streams is common.

The code generating these functions resides in media_sequence.h/.cc/.py and
media_sequence_util.h/.cc/.py. The media_sequence files generally defines the
API that should be used directly by developers. The media_sequence_util files
provide the function generation code used to define new features. If you require
additional features not supplied in the media_sequence files, use the functions
in media_sequence_util to create more in the appropriate namespace / module_dict
in your own files and import those as well.

In these prototypes, the prefix is optional as indicated by \[ \]s. The C++
types are abbreviated. The code and test cases are recommended for understanding
the exact types. The purpose of these example is to provide an illustration of
the pattern.

### Singular Context Features

| python call | c++ call | description |
|-------------|----------|-------------|
|`has_feature(example [, prefix])`|`HasFeature([const string& prefix,] const tf::SE& example)`|Returns a boolean if the feature is present.|
|`get_feature(example [, prefix])`|`GetFeature([const string& prefix,] const tf::SE& example)`|Returns a single feature of the appropriate type (string, int64, float).|
|`clear_feature(example [, prefix])`|`ClearFeature([const string& prefix,] tf::SE* example)`|Clears the feature.|
|`set_feature(value, example [, prefix])`|`SetFeature([const string& prefix,], const TYPE& value, tf::SE* example)`|Clears and stores the feature of the appropriate type.|
|`get_feature_key([prefix])`|`GetFeatureKey([const string& prefix])`|Returns the key used by related functions.|
|`get_feature_default_parser()`| | Returns the tf.io.FixedLenFeature for this type. (Python only.) |

### List Context Features

| python call | c++ call | description |
|-------------|----------|-------------|
|`has_feature(example [, prefix])`|`HasFeature([const string& prefix,] const tf::SE& example)`|Returns a boolean if the feature is present.|
|`get_feature(example [, prefix])`|`GetFeature([const string& prefix,] const tf::SE& example)`|Returns a sequence feature of the appropriate type (comparable to list/vector of string, int64, float).|
|`clear_feature(example [, prefix])`|`ClearFeature([const string& prefix,] tf::SE* example)`|Clears the feature.|
|`set_feature(values, example [, prefix])`|`SetFeature([const string& prefix,], const vector<TYPE>& values, tf::SE* example)`|Clears and stores the list of features of the appropriate type.|
|`get_feature_key([prefix])`|`GetFeatureKey([const string& prefix])`|Returns the key used by related functions.|
|`get_feature_default_parser()`| | Returns the tf.io.VarLenFeature for this type. (Python only.) |

### Singular Feature Lists

| python call | c++ call | description |
|-------------|----------|-------------|
|`has_feature(example [, prefix])`|`HasFeature([const string& prefix,] const tf::SE& example)`|Returns a boolean if the feature is present.|
|`get_feature_size(example [, prefix])`|`GetFeatureSize([const string& prefix,] const tf::SE&(example)`|Returns the number of features under this key. Will be 0 if the feature is absent.|
|`get_feature_at(index, example [, prefix])`|`GetFeatureAt([const string& prefix,] const tf::SE& example, const int index)`|Returns a single feature of the appropriate type (string, int64, float) at position index of the feature list.|
|`clear_feature(example [, prefix])`|`ClearFeature([const string& prefix,] tf::SE* example)`|Clears the entire feature.|
|`add_feature(value, example [, prefix])`|`AddFeature([const string& prefix,], const TYPE& value, tf::SE* example)`|Appends a feature of the appropriate type to the feature list.|
|`get_feature_key([prefix])`|`GetFeatureKey([const string& prefix])`|Returns the key used by related functions.|
|`get_feature_default_parser()`| | Returns the tf.io.FixedLenSequenceFeature for this type. (Python only.) |

### List Feature Lists

| python call | c++ call | description |
|-------------|----------|-------------|
|`has_feature(example [, prefix])`|`HasFeature([const string& prefix,] const tf::SE& example)`|Returns a boolean if the feature is present.|
|`get_feature_size(example [, prefix])`|`GetFeatureSize([const string& prefix,] const tf::SE& example)`|Returns the number of feature sequences under this key. Will be 0 if the feature is absent.|
|`get_feature_at(index, example [, prefix])`|`GetFeatureAt([const string& prefix,] const tf::SE& example, const int index)`|Returns a repeated feature of the appropriate type (comparable to list/vector of string, int64, float) at position index of the feature list.|
|`clear_feature(example [, prefix])`|`ClearFeature([const string& prefix,] tf::SE* example)`|Clears the entire feature.|
|`add_feature(value, example [, prefix])`|`AddFeature([const string& prefix,], const vector<TYPE>& value, tf::SE* example)`|Appends a sequence of features of the appropriate type to the feature list.|
|`get_feature_key([prefix])`|`GetFeatureKey([const string& prefix])`|Returns the key used by related functions.|
|`get_feature_default_parser()`| | Returns the tf.io.VarLenFeature for this type. (Python only.) |


## Keys

These keys are broadly useful for covering the range of multimedia based machine
learning tasks. The key itself should be human interpretable, and descriptions
are provided for elaboration.

### Keys related to the entire example
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`example/id`|context bytes|`set_example_id` / `SetExampleId`|A unique identifier for each example.|
|`example/dataset_name`|context bytes|`set_example_dataset_name` / `SetExampleDatasetName`|The name of the data set, including the version.|
|`example/dataset/flag/string`|context bytes list|`set_example_dataset_flag_string` / `SetExampleDatasetFlagString`|A list of bytes for dataset related attributes or flags for this example.

### Keys related to a clip
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`clip/data_path`|context bytes|`set_clip_data_path` / `SetClipDataPath`|The relative path to the data on disk from some root directory.|
|`clip/start/timestamp`|context int|`set_clip_start_timestamp` / `SetClipStartTimestamp`|The start time, in microseconds, for the start of the clip in the media.|
|`clip/end/timestamp`|context int|`set_clip_end_timestamp` / `SetClipEndTimestamp`|The end time, in microseconds, for the end of the clip in the media.|
|`clip/label/index`|context int list|`set_clip_label_index` / `SetClipLabelIndex`|A list of label indices for this clip.|
|`clip/label/string`|context string list|`set_clip_label_string` / `SetClipLabelString`|A list of label strings for this clip.|
|`clip/label/confidence`|context float list|`set_clip_label_confidence` / `SetClipLabelConfidence`|A list of label confidences for this clip.|
|`clip/media_id`|context bytes|`set_clip_media_id` / `SetClipMediaId`|Any identifier for the media beyond the data path.|
|`clip/alternative_media_id`|context bytes|`set_clip_alternative_media_id` / `SetClipAlternativeMediaId`|Yet another alternative identifier.|
|`clip/encoded_media_bytes`|context bytes|`set_clip_encoded_media_bytes` / `SetClipEncodedMediaBytes`|The encoded bytes for storing media directly in the SequenceExample.|
|`clip/encoded_media_start_timestamp`|context int|`set_clip_encoded_media_start_timestamp` / `SetClipEncodedMediaStartTimestamp`|The start time for the encoded media if not preserved during encoding.|

### Keys related to segments of clips
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`segment/start/timestamp`|context int list|`set_segment_start_timestamp` / `SetSegmentStartTimestamp`|A list of segment start times in microseconds.|
|`segment/start/index`|context int list|`set_segment_start_index` / `SetSegmentstartIndex`|A list of indices marking the first frame index >= the start time.|
|`segment/end/timestamp`|context int list|`set_segment_end_timestamp` / `SetSegmentEndTimestamp`|A list of segment end times in microseconds.|
|`segment/end/index`|context int list|`set_segment_end_index` / `SetSegmentEndIndex`|A list of indices marking the last frame index <= the end time.|
|`segment/label/index`|context int list|`set_segment_label_index` / `SetSegmentLabelIndex`|A list with the label index for each segment. Multiple labels for the same segment are encoded as repeated segments.|
|`segment/label/string`|context bytes list|`set_segment_label_string` / `SetSegmentLabelString`|A list with the label string for each segment. Multiple labels for the same segment are encoded as repeated segments.|
|`segment/label/confidence`|context float list|`set_segment_label_confidence` / `SetSegmentLabelConfidence`|A list with the label confidence for each segment. Multiple labels for the same segment are encoded as repeated segments.|

### Keys related to spatial regions (e.g. bounding boxes)
Prefixes are used to distinguish betwen different semantic meanings of regions.
This practice is so common, that the BBox version of function calls will be
provided. Each call accepts an optional prefix to avoid name collisions.
"Region" is used in the keys because of the similar semantic meaning between
different types of regions.

A few *special* accessors are provided to work with multiple keys at once.

Regions can be given identifiers for labels, tracks, and classes. Although
similar information can be stored in each identifier, the intended use is
different. Labels should be used when predicting a label for a region (such as
the class of the bounding box or action performed by a person). Tracks should be
used to uniquely identify regions over sequential frames. Classes are only
intended to be used to disambiguate track ids if those ids are not unique across
object labels. The recommendation is to prefer label fields for classification
tasks and tracking (or class) fields for tracking information.

| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`region/bbox/ymin`|feature list float list|`add_bbox_ymin` / `AddBBoxYMin`|A list of normalized minimum y values of bounding boxes in a frame.|
|`region/bbox/xmin`|feature list float list|`add_bbox_xmin` / `AddBBoxXMin`|A list of normalized minimum x values of bounding boxes in a frame.|
|`region/bbox/ymax`|feature list float list|`add_bbox_ymax` / `AddBBoxYMax`|A list of normalized maximum y values of bounding boxes in a frame.|
|`region/bbox/xmax`|feature list float list|`add_bbox_xmax` / `AddBBoxXMax`|A list of normalized maximum x values of bounding boxes in a frame.|
|`region/bbox/\*`| *special* |`add_bbox` / `AddBBox`|Operates on ymin,xmin,ymax,xmax with a single call.|
|`region/point/x`|feature list float list|`add_bbox_point_x` / `AddBBoxPointX`|A list of normalized x values for points in a frame.|
|`region/point/y`|feature list float list|`add_bbox_point_y` / `AddBBoxPointY`|A list of normalized y values for points in a frame.|
|`region/point/\*`| *special* |`add_bbox_point` / `AddBBoxPoint`|Operates on point/x,point/y with a single call.|
|`region/radius`|feature list float list|`add_bbox_point_radius` / `AddBBoxRadius`|A list of radii for points in a frame.|
|`region/3d_point/x`|feature list float list|`add_bbox_3d_point_x` / `AddBBox3dPointX`|A list of normalized x values for points in a frame.|
|`region/3d_point/y`|feature list float list|`add_bbox_3d_point_y` / `AddBBox3dPointY`|A list of normalized y values for points in a frame.|
|`region/3d_point/z`|feature list float list|`add_bbox_3d_point_z` / `AddBBox3dPointZ`|A list of normalized z values for points in a frame.|
|`region/3d_point/\*`| *special* |`add_bbox_3d_point` / `AddBBox3dPoint`|Operates on 3d_point/{x,y,z} with a single call.|
|`region/timestamp`|feature list int|`add_bbox_timestamp` / `AddBBoxTimestamp`|The timestamp in microseconds for the region annotations.|
|`region/num_regions`|feature list int|`add_bbox_num_regions` / `AddBBoxNumRegions`|The number of boxes or other regions in a frame. Should be 0 for unannotated frames.|
|`region/is_annotated`|feature list int|`add_bbox_is_annotated` / `AddBBoxIsAnnotated`|1 if this timestep is annotated. 0 otherwise. Distinguishes empty from unannotated frames.|
|`region/is_generated`|feature list int list|`add_bbox_is_generated` / `AddBBoxIsGenerated`|For each region, 1 if the region is procedurally generated for this frame.|
|`region/is_occluded`|feature list int list|`add_bbox_is_occluded` / `AddBBoxIsOccluded`|For each region, 1 if the region is occluded in the current frame.|
|`region/label/index`|feature list int list|`add_bbox_label_index` / `AddBBoxLabelIndex`|For each region, lists the integer label. Multiple labels for one region require duplicating the region.|
|`region/label/string`|feature list bytes list|`add_bbox_label_string` / `AddBBoxLabelString`|For each region, lists the string label. Multiple labels for one region require duplicating the region.|
|`region/label/confidence`|feature list float list|`add_bbox_label_confidence` / `AddBBoxLabelConfidence`|For each region, lists the confidence or weight for the label. Multiple labels for one region require duplicating the region.|
|`region/track/index`|feature list int list|`add_bbox_track_index` / `AddBBoxTrackIndex`|For each region, lists the integer track id. Multiple track ids for one region require duplicating the region.|
|`region/track/string`|feature list bytes list|`add_bbox_track_string` / `AddBBoxTrackString`|For each region, lists the string track id. Multiple track ids for one region require duplicating the region.|
|`region/track/confidence`|feature list float list|`add_bbox_track_confidence` / `AddBBoxTrackConfidence`|For each region, lists the confidence or weight for the track. Multiple track ids for one region require duplicating the region.|
|`region/class/index`|feature list int list|`add_bbox_class_index` / `AddBBoxClassIndex`|For each region, lists the integer class. Multiple classes for one region require duplicating the region.|
|`region/class/string`|feature list bytes list|`add_bbox_class_string` / `AddBBoxClassString`|For each region, lists the string class. Multiple classes for one region require duplicating the region.|
|`region/class/confidence`|feature list float list|`add_bbox_class_confidence` / `AddBBoxClassConfidence`|For each region, lists the confidence or weight for the class. Multiple classes for one region require duplicating the region.|
|`region/embedding/float`|feature list float list|`add_bbox_embedding_floats` / `AddBBoxEmbeddingFloats`|For each region, provide an embedding as sequence of floats.|
|`region/parts`|context bytes list|`set_bbox_parts` / `SetBBoxParts`|The list of region parts expected in this example.|
|`region/embedding/ dimensions_per_region`|context int list|`set_bbox_embedding_dimensions_per_region` / `SetBBoxEmbeddingDimensionsPerRegion`|Provide the dimensions for each embedding.|
|`region/embedding/format`|context string|`set_bbox_embedding_format` / `SetBBoxEmbeddingFormat`|Provides the encoding format, if any, for region embeddings.|
|`region/embedding/encoded`|feature list bytes list|`add_bbox_embedding_encoded` / `AddBBoxEmbeddingEncoded`|For each region, provide an encoded embedding.|
|`region/embedding/confidence`|feature list float list|`add_bbox_embedding_confidence` / `AddBBoxEmbeddingConfidence` | For each region, provide a confidence for the embedding.|
|`region/unmodified_timestamp`|feature list int|`add_bbox_unmodified_timestamp` / `AddUnmodifiedBBoxTimestamp`|Used to store the original timestamps if procedurally aligning timestamps to image frames.|

### Keys related to images
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`image/encoded`|feature list bytes|`add_image_encoded` / `AddImageEncoded`|The encoded image at each timestep.|
|`image/timestamp`|feature list int|`add_image_timestamp` / `AddImageTimestamp`|The timestamp in microseconds for the image.|
|`image/multi_encoded`|feature list bytes list|`add_image_multi_encoded` / `AddImageMultiEncoded`|Storing multiple images at each timestep (e.g. from multiple camera views).|
|`image/label/index`|feature list int list|`add_image_label_index` / `AddImageLabelIndex`|If an image at a specific timestamp should have a label, use this. If a range of time, prefer Segments instead.|
|`image/label/string`|feature list bytes list|`add_image_label_string` / `AddImageLabelString`|If an image at a specific timestamp should have a label, use this. If a range of time, prefer Segments instead.|
|`image/label/confidence`|feature list float list|`add_image_label_confidence` / `AddImageLabelConfidence`|If an image at a specific timestamp should have a label, use this. If a range of time, prefer Segments instead.|
|`image/format`|context bytes|`set_image_format` / `SetImageFormat`|The encoding format of the images.|
|`image/channels`|context int|`set_image_channels` / `SetImageChannels`|The number of channels in the image.|
|`image/colorspace`|context bytes|`set_image_colorspace` / `SetColorspace`|The colorspace of the images.|
|`image/height`|context int|`set_image_height` / `SetImageHeight`|The height of the image in pixels.|
|`image/width`|context int|`set_image_width` / `SetImageWidth`|The width of the image in pixels.|
|`image/frame_rate`|context float|`set_image_frame_rate` / `SetImageFrameRate`|The rate of images in frames per second.|
|`image/data_path`|context bytes|`set_image_data_path` / `SetImageDataPath`|The path of the image file if it did not come from a media clip.|

### Keys related to image class segmentation
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`CLASS_SEGMENTATION/image/encoded`|feature list bytes|`add_class_segmentation_encoded` / `AddClassSegmentationEncoded`|The encoded image of class labels at each timestep.|
|`CLASS_SEGMENTATION/image/timestamp`|feature list int|`add_class_segmentation_timestamp` / `AddClassSegmentationTimestamp`|The timestamp in microseconds for the class labels.|
|`CLASS_SEGMENTATION/image/multi_encoded`|feature list bytes list|`add_class_segmentation_multi_encoded` / `AddClassSegmentationMultiEncoded`|Storing multiple segmentation masks in case they overlap.|
|`CLASS_SEGMENTATION/image/format`|context bytes|`set_class_segmentation_format` / `SetClassSegmentationFormat`|The encoding format of the class label images.|
|`CLASS_SEGMENTATION/image/height`|context int|`set_class_segmentation_height` / `SetClassSegmentationHeight`|The height of the image in pixels.|
|`CLASS_SEGMENTATION/image/width`|context int|`set_class_segmentation_width` / `SetClassSegmentationWidth`|The width of the image in pixels.|
|`CLASS_SEGMENTATION/image/class/ label/index`|context int list|`set_class_segmentation_class_label_index` / `SetClassSegmentationClassLabelIndex`|If necessary a mapping from values in the image to class labels.|
|`CLASS_SEGMENTATION/image/class/ label/string`|context bytes list|`set_class_segmentation_class_label_string` / `SetClassSegmentationClassLabelString`|A mapping from values in the image to class labels.|

### Keys related to image instance segmentation
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`INSTANCE_SEGMENTATION/image/ encoded`|feature list bytes|`add_instance_segmentation_encoded` / `AddInstanceSegmentationEncoded`|The encoded image of object instance labels at each timestep.|
|`INSTANCE_SEGMENTATION/image/ timestamp`|feature list int|`add_instance_segmentation_timestamp` / `AddInstanceSegmentationTimestamp`|The timestamp in microseconds for the object instance labels.|
|`INSTANCE_SEGMENTATION/image/multi_encoded`|feature list bytes list|`add_instance_segmentation_multi_encoded` / `AddInstanceSegmentationEncoded`|Storing multiple segmentation masks in case they overlap.|
|`INSTANCE_SEGMENTATION/image/ format`|context bytes|`set_instance_segmentation_format` / `SetInstanceSegmentationFormat`|The encoding format of the object instance labels.|
|`INSTANCE_SEGMENTATION/image/ height`|context int|`set_instance_segmentation_height` / `SetInstanceSegmentationHeight`|The height of the image in pixels.|
|`INSTANCE_SEGMENTATION/image/ width`|context int|`set_instance_segmentation_width` / `SetInstanceSegmentationWidth`|The width of the image in pixels.|
|`INSTANCE_SEGMENTATION/image/ class/label/index`|context int list|`set_instance_segmentation_class_label_index` / `SetInstanceSegmentationClassLabelIndex`|If necessary a mapping from values in the image to class labels.|
|`INSTANCE_SEGMENTATION/image/ class/label/string`|context bytes list|`set_instance_segmentation_class_label_string` / `SetInstanceSegmentationClassLabelString`|A mapping from values in the image to class labels.|
|`INSTANCE_SEGMENTATION/image/ object/class/index`|context int|`set_instance_segmentation_object_class_index` / `SetInstanceSegmentationObjectClassIndex`|If necessary a mapping from values in the image to class indices.|

### Keys related to optical flow
| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`FORWARD_FLOW/image/encoded`|feature list bytes|`add_forward_flow_encoded` / `AddForwardFlowEncoded`|The encoded forward optical flow field at each timestep.|
|`FORWARD_FLOW/image/timestamp`|feature list int|`add_forward_flow_timestamp` / `AddForwardFlowTimestamp`|The timestamp in microseconds for the optical flow field.|
|`FORWARD_FLOW/image/multi_encoded`|feature list bytes list|`add_forward_flow_multi_encoded` / `AddForwardFlowMultiEncoded`|Storing multiple optical flow fields at each timestep (e.g. from multiple camera views).|
|`FORWARD_FLOW/image/format`|context bytes|`set_forward_flow_format` / `SetForwardFlowFormat`|The encoding format of the optical flow field.|
|`FORWARD_FLOW/image/channels`|context int|`set_forward_flow_channels` / `SetForwardFlowChannels`|The number of channels in the optical flow field.|
|`FORWARD_FLOW/image/height`|context int|`set_forward_flow_height` / `SetForwardFlowHeight`|The height of the optical flow field in pixels.|
|`FORWARD_FLOW/image/width`|context int|`set_forward_flow_width` / `SetForwardFlowWidth`|The width of the optical flow field in pixels.|
|`FORWARD_FLOW/image/frame_rate`|context float|`set_forward_flow_frame_rate` / `SetForwardFlowFrameRate`|The rate of optical flow field in frames per second.|
|`FORWARD_FLOW/image/saturation`|context float|`set_forward_flow_saturation` / `SetForwardFlowSaturation`|The saturation value used before encoding the flow field to an image.|


### Keys related to generic features
Storing generic features is powerful, but potentially confusing. The
recommendation is to use more specific methods if possible. When using these
generic features, always supply a prefix. (The recommended prefix format,
enforced by some MediaPipe functions, is all caps with underscores, e.g.
MY_FAVORITE_FEATURE.) Following this recommendation, the keys will be listed
with a generic PREFIX. Calls exist for storing generic features in both the
`feature_list` and the `context`. For anything that occurs with a timestamp,
use the `feature_list`; for anything that applies to the example as a whole,
without timestamps, use the `context`.

| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`PREFIX/feature/floats`|feature list float list|`add_feature_floats` / `AddFeatureFloats`|A list of floats at a timestep.|
|`PREFIX/feature/bytes`|feature list bytes list|`add_feature_bytes` / `AddFeatureBytes`|A list of bytes at a timestep. Maybe be encoded.|
|`PREFIX/feature/ints`|feature list int list|`add_feature_ints` / `AddFeatureInts`|A list of ints at a timestep.|
|`PREFIX/feature/timestamp`|feature list int|`add_feature_timestamp` / `AddFeatureTimestamp`|A timestamp for a set of features.|
|`PREFIX/feature/duration`|feature list int list|`add_feature_duration` / `AddFeatureDuration`|It is occasionally useful to indicate that a feature applies to a time range. This should only be used for features and annotations should be provided as Segments.|
|`PREFIX/feature/confidence`|feautre list float list|`add_feature_confidence` / `AddFeatureConfidence`|The confidence for each generated feature.|
|`PREFIX/feature/dimensions`|context int list|`set_feature_dimensions` / `SetFeatureDimensions`|A list of integer dimensions for each feature.|
|`PREFIX/feature/rate`|context float|`set_feature_rate` / `SetFeatureRate`|The rate that features are calculated as features per second.|
|`PREFIX/feature/bytes/format`|context bytes|`set_feature_bytes_format` / `SetFeatureBytesFormat`|The encoding format if any for features stored as bytes.|
|`PREFIX/context_feature/floats`|context float list|`add_context_feature_floats` / `AddContextFeatureFloats`|A list of floats for the entire example.|
|`PREFIX/context_feature/bytes`|context bytes list|`add_context_feature_bytes` / `AddContextFeatureBytes`|A list of bytes for the entire example. Maybe be encoded.|
|`PREFIX/context_feature/ints`|context int list|`add_context_feature_ints` / `AddContextFeatureInts`|A list of ints for the entire example.|

### Keys related to audio
Audio is a special subtype of generic features with additional data about the
audio format. When using audio, always supply a prefix. The keys here will be
listed with a generic PREFIX.

To understand the terminology, it is helpful conceptualize the audio as a list
of matrices. The columns of the matrix are called samples. The rows of the
matrix are called channels. Each matrix is called a packet. The packet rate is
how often packets appear per second. The sample rate is the rate of columns per
second. The audio sample rate is used for derived features such as spectrograms
where the STFT is computed over audio at some other rate.

| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`PREFIX/feature/floats`|feature list float list|`add_feature_floats` / `AddFeatureFloats`|A list of floats at a timestep.|
|`PREFIX/feature/timestamp`|feature list int|`add_feature_timestamp` / `AddFeatureTimestamp`|A timestamp for a set of features.|
|`PREFIX/feature/sample_rate`|context float|`set_feature_sample_rate` / `SetFeatureSampleRate`|The number of features per second. (e.g. for a spectrogram, this is the rate of STFT windows.)|
|`PREFIX/feature/num_channels`|context int|`set_feature_num_channels` / `SetFeatureNumChannels`|The number of channels of audio in each stored feature.|
|`PREFIX/feature/num_samples`|context int|`set_feature_num_samples` / `SetFeatureNumSamples`|The number of samples of audio in each stored feature.|
|`PREFIX/feature/packet_rate`|context float|`set_feature_packet_rate` / `SetFeaturePacketRate`|The number of packets per second.|
|`PREFIX/feature/audio_sample_rate`|context float|`set_feature_audio_sample_rate` / `SetFeatureAudioSampleRate`|The sample rate of the original audio for derived features.|

### Keys related to text, captions, and ASR
Text features may be timed with the media such as captions or automatic
speech recognition results, or may be descriptions. This collection of keys
should be used for many, very short text features. For a few, longer segments
please use the Segment keys in the context as described above. As always,
prefixes can be used to store different types of text such as automated and
ground truth transcripts.

| key | type | python call / c++ call | description |
|-----|------|------------------------|-------------|
|`text/language`|context bytes|`set_text_langage` / `SetTextLanguage`|The language for the corresponding text.|
|`text/context/content`|context bytes|`set_text_context_content` / `SetTextContextContent`|Storage for large blocks of text in the context.|
|`text/content`|feature list bytes|`add_text_content` / `AddTextContent`|One (or a few) text tokens that occur at one timestamp.|
|`text/timestamp`|feature list int|`add_text_timestamp` / `AddTextTimestamp`|When a text token occurs in microseconds.|
|`text/duration`|feature list int|`add_text_duration` / `SetTextDuration`|The duration in microseconds for the corresponding text tokens.|
|`text/confidence`|feature list float|`add_text_confidence` / `AddTextConfidence`|How likely the text is correct.|
|`text/embedding`|feautre list float list|`add_text_embedding` / `AddTextEmbedding`|A floating point vector for the corresponding text token.|
|`text/token/id`|feature list int|`add_text_token_id` / `AddTextTokenId`|An integer id for the corresponding text token.|
