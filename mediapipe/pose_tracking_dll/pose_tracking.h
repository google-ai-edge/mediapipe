/**
   Copyright 2022, Nimagna AG

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#ifndef POSE_TRACKING_LIBRARY_H
#define POSE_TRACKING_LIBRARY_H

#ifdef COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

class PoseTrackingImpl;

namespace nimagna {
namespace cv_wrapper {
struct Point2f {
  float x = 0;
  float y = 0;

  Point2f() = default;
  Point2f(float x, float y) : x(x), y(y) {}
};
struct Point3f {
  float x = 0;
  float y = 0;
  float z = 0;

  Point3f() = default;
  Point3f(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct Rect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;

  Rect() = default;
  Rect(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}
};

struct Mat {
  int rows = 0;
  int cols = 0;
  unsigned char* data = 0;

  Mat(int rows, int cols, unsigned char* data) : rows(rows), cols(cols), data(data) {}
};
}  // namespace cv_wrapper

class DLLEXPORT PoseTracking {
 public:
  static constexpr size_t kLandmarksCount = 33u;
  enum LandmarkNames {
    NOSE = 0,
    LEFT_EYE_INNER,
    LEFT_EYE,
    LEFT_EYE_OUTER,
    RIGHT_EYE_INNER,
    RIGHT_EYE,
    RIGHT_EYE_OUTER,
    LEFT_EAR,
    RIGHT_EAR,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_PINKY,
    RIGHT_PINKY,
    LEFT_INDEX,
    RIGHT_INDEX,
    LEFT_THUMB,
    RIGHT_THUMB,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
    COUNT = kLandmarksCount
  };

  PoseTracking(const char* calculatorGraphConfigFile);
  ~PoseTracking();

  bool processFrame(const cv_wrapper::Mat& inputRGB8Bit);
  cv_wrapper::Mat lastSegmentedFrame();
  cv_wrapper::Point3f* lastDetectedLandmarks();

 private:
  PoseTrackingImpl* mImplementation;
};
}  // namespace nimagna

#endif
