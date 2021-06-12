// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_PIECEWISE_LINEAR_FUNCTION_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_PIECEWISE_LINEAR_FUNCTION_H_

#include <vector>

namespace mediapipe {
namespace autoflip {

// Implementation of piecewise linear functions. The function is specified as a
// series of points (x1,y1), (x2,y2),..., (xn,yn). It can be constructed
// programmatically by repeatedly calling the AddPoint(x, y) method.
class PiecewiseLinearFunction {
 public:
  PiecewiseLinearFunction() {}

  // Evaluate the function at the specified input. The output
  // saturates at the values of the first and last interpolation
  // points.
  // f(x) = y1 for x <= x1   // Saturate at the lowest value
  // f(x) = yn for x >  xn   // Saturate at the highest value
  // f(x) = (x-xj)/(xk-xj)*(yk-yj) + yk for xj < x <= xk and k = j+1
  double Evaluate(double input) const;

  // Adds the given point to the function.  Points must be added in
  // non-decreasing x order.  Because the points are given in sorted
  // order, this function can be used to construct discontinuous
  // functions.  For example, if one defines
  //    f.AddPoint(-1.0, 0.0)
  //    f.AddPoint( 0.0, 0.0)
  //    f.AddPoint( 0.0, 1.0)
  //    f.AddPoint( 1.0, 1.0)
  // the result function f is discontinuous at 0.0.  By convention,
  // this function will return f.Evaluate(0.0) = 0.0, and
  // f.Evaluate(1e-12) = 1.0.  This convention corresponds to the
  // natural behavior of GetIntervalIterator().
  void AddPoint(double x, double y);

 private:
  struct Point {
    double x;
    double y;
    Point(double X, double Y) : x(X), y(Y) {}
  };

  // A functor for use with stl algorithms like sort() and lower_bound() that
  // sorts by the point's x value.
  class PointCompare {
   public:
    bool operator()(const Point& p1, const Point& p2) const {
      return p1.x < p2.x;
    }
  };

  // Returns the iterator, i, closest to points_.begin() such that
  // input <= i->x or it returns points_.end() if input > all x values
  // in points_.
  std::vector<Point>::const_iterator GetIntervalIterator(double input) const;

  // Given two points p1 and p2 such that p1.x < input and p2.x >= input this
  // returns the linear interpolation of the y value.
  double Interpolate(const Point& p1, const Point& p2, double input) const;

  std::vector<Point> points_;
};

}  // namespace autoflip
}  // namespace mediapipe
#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_PIECEWISE_LINEAR_FUNCTION_H_
