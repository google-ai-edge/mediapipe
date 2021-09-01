#include "mediapipe/framework/formats/affine_transform.h"

#include <string>

#include "base/logging.h"
#include "mediapipe/framework/formats/affine_transform_data.pb.h"
#include "mediapipe/framework/port/point2.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {

TEST(AffineTransformTest, TraslationTest) {
  AffineTransform transform;
  transform.SetTranslation(Point2_f(10, -3));

  auto trans = transform.GetTranslation();
  EXPECT_FLOAT_EQ(10, trans.x());
  EXPECT_FLOAT_EQ(-3, trans.y());

  transform.AddTranslation(Point2_f(-10, 3));

  trans = transform.GetTranslation();
  EXPECT_FLOAT_EQ(0, trans.x());
  EXPECT_FLOAT_EQ(0, trans.y());
}

TEST(AffineTransformTest, ScaleTest) {
  AffineTransform transform;
  transform.SetScale(Point2_f(10, -3));

  auto scale = transform.GetScale();
  EXPECT_FLOAT_EQ(10, scale.x());
  EXPECT_FLOAT_EQ(-3, scale.y());

  transform.AddScale(Point2_f(-10, 3));

  scale = transform.GetScale();
  EXPECT_FLOAT_EQ(0, scale.x());
  EXPECT_FLOAT_EQ(0, scale.y());
}

TEST(AffineTransformTest, RotationTest) {
  AffineTransform transform;
  transform.SetRotation(0.7);

  float rot = transform.GetRotation();
  EXPECT_FLOAT_EQ(0.7, rot);

  transform.AddRotation(-0.7);
  rot = transform.GetRotation();
  EXPECT_FLOAT_EQ(0, rot);
}

TEST(AffineTransformTest, ShearTest) {
  AffineTransform transform;
  transform.SetShear(Point2_f(10, -3));

  auto shear = transform.GetShear();
  EXPECT_FLOAT_EQ(10, shear.x());
  EXPECT_FLOAT_EQ(-3, shear.y());

  transform.AddShear(Point2_f(-10, 3));

  shear = transform.GetShear();
  EXPECT_FLOAT_EQ(0, shear.x());
  EXPECT_FLOAT_EQ(0, shear.y());
}

TEST(AffineTransformTest, TransformTest) {
  AffineTransform transform1;
  transform1 = AffineTransform::Create(Point2_f(0.1, -0.2), Point2_f(0.3, -0.4),
                                       0.5, Point2_f(0.6, -0.7));

  AffineTransform transform2;
  transform2 = AffineTransform::Create(Point2_f(0.1, -0.2), Point2_f(0.3, -0.4),
                                       0.5, Point2_f(0.6, -0.7));

  EXPECT_THAT(true, transform1.Equals(transform2));
  EXPECT_THAT(true, AffineTransform::Equal(transform1, transform2));

  transform1 = AffineTransform::Create(Point2_f(0.00001, -0.00002),
                                       Point2_f(0.00003, -0.00004), 0.00005,
                                       Point2_f(0.00006, -0.00007));

  transform2 = AffineTransform::Create(Point2_f(0.00001, -0.00002),
                                       Point2_f(0.00003, -0.00004), 0.00005,
                                       Point2_f(0.00006, -0.00007));

  EXPECT_THAT(true, transform1.Equals(transform2, 0.000001));
  EXPECT_THAT(true, AffineTransform::Equal(transform1, transform2, 0.000001));
}

}  // namespace mediapipe
