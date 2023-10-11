---
layout: forward
target: https://developers.google.com/mediapipe/solutions/guide#legacy
title: Instant Motion Tracking
parent: MediaPipe Legacy Solutions
nav_order: 11
---

# MediaPipe Instant Motion Tracking
{: .no_toc }

<details close markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>
---

**Attention:** *Thank you for your interest in MediaPipe Solutions.
We have ended support for this MediaPipe Legacy Solution as of March 1, 2023.
For more information, see the
[MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/guide#legacy)
site.*

----

## Overview

Augmented Reality (AR) technology creates fun, engaging, and immersive user
experiences. The ability to perform AR tracking across devices and platforms,
without initialization, remains important to power AR applications at scale.

MediaPipe Instant Motion Tracking provides AR tracking across devices and
platforms without initialization or calibration. It is built upon the
[MediaPipe Box Tracking](./box_tracking.md) solution. With Instant Motion
Tracking, you can easily place virtual 2D and 3D content on static or moving
surfaces, allowing them to seamlessly interact with the real-world environment.

![instant_motion_tracking_android_small](https://mediapipe.dev/images/mobile/instant_motion_tracking_android_small.gif) |
:-----------------------------------------------------------------------: |
*Fig 1. Instant Motion Tracking is used to augment the world with a 3D sticker.* |

## Pipeline

The Instant Motion Tracking pipeline is implemented as a MediaPipe
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/instant_motion_tracking.pbtxt),
which internally utilizes a
[RegionTrackingSubgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/subgraphs/region_tracking.pbtxt)
in order to perform anchor tracking for each individual 3D sticker.

We first use a
[StickerManagerCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/calculators/sticker_manager_calculator.cc)
to prepare the individual sticker data for the rest of the application. This
information is then sent to the
[RegionTrackingSubgraph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/subgraphs/region_tracking.pbtxt)
that performs 3D region tracking for sticker placement and rendering. Once
acquired, our tracked sticker regions are sent with user transformations (i.e.
gestures from the user to rotate and zoom the sticker) and IMU data to the
[MatricesManagerCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/calculators/matrices_manager_calculator.cc),
which turns all our sticker transformation data into a set of model matrices.
This data is handled directly by our
[GlAnimationOverlayCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/calculators/gl_animation_overlay_calculator.cc)
as an input stream, which will render the provided texture and object file using
our matrix specifications. The output of
[GlAnimationOverlayCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/calculators/gl_animation_overlay_calculator.cc)
is a video stream depicting the virtual 3D content rendered on top of the real
world, creating immersive AR experiences for users.

## Using Instant Motion Tracking

With the Instant Motion Tracking MediaPipe [graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/instant_motion_tracking.pbtxt),
an application can create an interactive and realistic AR experience by
specifying the required input streams, side packets, and output streams.
The input streams are the following:

*    Input Video (GpuBuffer): Video frames to render augmented stickers onto.
*    Rotation Matrix (9-element Float Array): The 3x3 row-major rotation
matrix from the device IMU to determine proper orientation of the device.
*    Sticker Proto String (String): A string representing the
serialized [sticker buffer protobuf message](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/calculators/sticker_buffer.proto),
containing a list of all stickers and their attributes.
  *    Each sticker in the Protobuffer has a unique ID to find associated
  anchors and transforms, an initial anchor placement in a normalized [0.0, 1.0]
  3D space, a user rotation and user scaling transform on the sticker,
  and an integer indicating which type of objects to render for the
  sticker (e.g. 3D asset or GIF).
*    Sticker Sentinel (Integer): When an anchor must be initially placed or
repositioned, this value must be changed to the ID of the anchor to reset from
the sticker buffer protobuf message. If no valid ID is provided, the system
will simply maintain tracking.

Side packets are also an integral part of the Instant Motion Tracking solution
to provide device-specific information for the rendering system:

*   Field of View (Float): The field of view of the camera in radians.
*   Aspect Ratio (Float): The aspect ratio (width / height) of the camera frames
    (this ratio corresponds to the image frames themselves, not necessarily the
    screen bounds).
*   Object Asset (String): The
    [GlAnimationOverlayCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/calculators/gl_animation_overlay_calculator.cc)
    must be provided with an associated asset file name pointing to the 3D model
    to render in the viewfinder.
*   (Optional) Texture (ImageFrame on Android, GpuBuffer on iOS): Textures for
    the
    [GlAnimationOverlayCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/calculators/gl_animation_overlay_calculator.cc)
    can be provided either via an input stream (dynamic texturing) or as a side
    packet (unchanging texture).

The rendering system for the Instant Motion Tracking is powered by OpenGL. For
more information regarding the structure of model matrices and OpenGL rendering,
please visit [OpenGL Wiki](https://www.khronos.org/opengl/wiki/). With the
specifications above, the Instant Motion Tracking capabilities can be adapted to
any device that is able to run the MediaPipe framework with a working IMU system
and connected camera.

## Example Apps

Please first see general instructions for
[Android](../getting_started/android.md) on how to build MediaPipe examples.

* Graph: [mediapipe/graphs/instant_motion_tracking/instant_motion_tracking.pbtxt](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/instant_motion_tracking/instant_motion_tracking.pbtxt)

* Android target (or download prebuilt [ARM64 APK](https://drive.google.com/file/d/1KnaBBoKpCHR73nOBJ4fL_YdWVTAcwe6L/view?usp=sharing)):
[`mediapipe/examples/android/src/java/com/google/mediapipe/apps/instantmotiontracking:instantmotiontracking`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/instantmotiontracking/BUILD)

* Assets rendered by the [GlAnimationOverlayCalculator](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/calculators/gl_animation_overlay_calculator.cc) must be preprocessed into an OpenGL-ready custom .uuu format. This can be done
for user assets as follows:
> First run
>
> ```shell
> ./mediapipe/graphs/object_detection_3d/obj_parser/obj_cleanup.sh [INPUT_DIR] [INTERMEDIATE_OUTPUT_DIR]
> ```
> and then run
>
> ```build
> bazel run -c opt mediapipe/graphs/object_detection_3d/obj_parser:ObjParser -- input_dir=[INTERMEDIATE_OUTPUT_DIR] output_dir=[OUTPUT_DIR]
> ```
> INPUT_DIR should be the folder with initial asset .obj files to be processed,
> and OUTPUT_DIR is the folder where the processed asset .uuu file will be placed.
>
> Note: ObjParser combines all .obj files found in the given directory into a
> single .uuu animation file, using the order given by sorting the filenames alphanumerically. Also the ObjParser directory inputs must be given as
> absolute paths, not relative paths. See parser utility library at [`mediapipe/graphs/object_detection_3d/obj_parser/`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/obj_parser/) for more details.

## Resources

*   Google Developers Blog:
    [Instant Motion Tracking With MediaPipe](https://developers.googleblog.com/2020/08/instant-motion-tracking-with-mediapipe.html)
*   Google AI Blog:
    [The Instant Motion Tracking Behind Motion Stills AR](https://ai.googleblog.com/2018/02/the-instant-motion-tracking-behind.html)
*   Paper:
    [Instant Motion Tracking and Its Applications to Augmented Reality](https://arxiv.org/abs/1907.06796)
