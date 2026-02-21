#include <cstdint>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

TEST(TensorTest, DebugStringFloat32) {
  Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape{2, 2});
  {
    auto v = t.GetCpuWriteView();
    float* f = v.buffer<float>();
    f[0] = 1.0f;
    f[1] = 2.1f;
    f[2] = 3.2f;
    f[3] = 4.3f;
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<Float32> [2 2] =\n"
            "[[1   2.1]\n"
            " [3.2 4.3]]");
}

TEST(TensorTest, DebugStringInt32) {
  Tensor t(Tensor::ElementType::kInt32, Tensor::Shape{2, 3});
  {
    auto v = t.GetCpuWriteView();
    int32_t* i = v.buffer<int32_t>();
    for (int j = 0; j < 6; ++j) {
      i[j] = j + 1;
    }
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<Int32> [2 3] =\n"
            "[[1 2 3]\n"
            " [4 5 6]]");
}

TEST(TensorTest, DebugStringInt64) {
  Tensor t(Tensor::ElementType::kInt64, Tensor::Shape{2, 2});
  {
    auto v = t.GetCpuWriteView();
    int64_t* i = v.buffer<int64_t>();
    i[0] = 10000000000L;
    i[1] = 20000000000L;
    i[2] = 30000000000L;
    i[3] = 40000000000L;
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<Int64> [2 2] =\n"
            "[[10000000000 20000000000]\n"
            " [30000000000 40000000000]]");
}

TEST(TensorTest, DebugStringUInt8) {
  Tensor t(Tensor::ElementType::kUInt8, Tensor::Shape{4});
  {
    auto v = t.GetCpuWriteView();
    uint8_t* u = v.buffer<uint8_t>();
    u[0] = 10;
    u[1] = 20;
    u[2] = 30;
    u[3] = 40;
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<UInt8> [4] =\n"
            "[10 20 30 40]");
}

TEST(TensorTest, DebugStringBool) {
  Tensor t(Tensor::ElementType::kBool, Tensor::Shape{4});
  {
    auto v = t.GetCpuWriteView();
    bool* b = v.buffer<bool>();
    b[0] = true;
    b[1] = false;
    b[2] = true;
    b[3] = false;
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<Bool> [4] =\n"
            "[ true false  true false]");
}

TEST(TensorTest, DebugStringChar) {
  Tensor t(Tensor::ElementType::kChar, Tensor::Shape{7});
  {
    auto v = t.GetCpuWriteView();
    char* c = v.buffer<char>();
    c[0] = 'a';
    c[1] = 'b';
    c[2] = 'c';
    c[3] = '\t';
    c[4] = 'd';
    c[5] = 'e';
    c[6] = 0;
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<Char> [7] =\n"
            "[   a    b    c \\x09    d    e \\x00]");
}

TEST(TensorTest, DebugStringEmpty) {
  Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape{0});
  EXPECT_EQ(t.DebugString(),
            "Tensor<Float32> [0] =\n"
            "[]");
}

TEST(TensorTest, DebugStringHighDimension) {
  Tensor t(Tensor::ElementType::kInt32, Tensor::Shape{2, 1, 2, 2});
  {
    auto v = t.GetCpuWriteView();
    int32_t* i = v.buffer<int32_t>();
    for (int j = 0; j < 8; ++j) {
      i[j] = j;
    }
  }
  EXPECT_EQ(t.DebugString(),
            "Tensor<Int32> [2 1 2 2] =\n"
            "[[[[0 1]\n"
            "   [2 3]]]\n"
            "\n"
            "\n"
            " [[[4 5]\n"
            "   [6 7]]]]");
}

TEST(TensorTest, DebugStringShortenedHighDimension) {
  // 1024 elements, so the last dimension is shortened.
  Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape{64, 7, 9});
  {
    auto v = t.GetCpuWriteView();
    float* f = v.buffer<float>();
    for (int i = 0; i < 64 * 7 * 9; ++i) {
      f[i] = i * 0.123f;
    }
  }
  EXPECT_EQ(t.DebugString(/*max_num_elements=*/1024),
            R"(Tensor<Float32> [64 7 9] =
[[[  0          0.123      0.246    ...   0.738      0.861      0.984   ]
  [  1.107      1.23       1.353    ...   1.845      1.968      2.091   ]
  [  2.214      2.337      2.46     ...   2.952      3.075      3.198   ]
  [  3.321      3.444      3.567    ...   4.059      4.182      4.305   ]
  [  4.428      4.551      4.674    ...   5.166      5.289      5.412   ]
  [  5.535      5.658      5.781    ...   6.273      6.396      6.519   ]
  [  6.642      6.765      6.888    ...   7.38       7.503      7.626   ]]

 [[  7.749      7.872      7.995    ...   8.487      8.610001   8.733   ]
  [  8.856      8.979      9.102    ...   9.594      9.717      9.84    ]
  [  9.963     10.086     10.209    ...  10.701     10.824     10.947   ]
  [ 11.07      11.193     11.316    ...  11.808     11.931     12.054   ]
  [ 12.177     12.3       12.423    ...  12.915     13.038     13.161   ]
  [ 13.284     13.407     13.53     ...  14.022     14.145     14.268   ]
  [ 14.391     14.514     14.637    ...  15.129     15.252     15.375   ]]

 [[ 15.498     15.621     15.744    ...  16.236     16.359     16.482   ]
  [ 16.605     16.728     16.851    ...  17.343     17.466     17.589   ]
  [ 17.712     17.835     17.958    ...  18.45      18.573     18.696   ]
  [ 18.819     18.942     19.065    ...  19.557     19.68      19.803   ]
  [ 19.926     20.049     20.172    ...  20.664     20.787     20.91    ]
  [ 21.033     21.156     21.279    ...  21.771     21.894     22.017   ]
  [ 22.14      22.263     22.386    ...  22.878     23.001     23.124   ]]

 ...

 [[472.689    472.812    472.935    ... 473.427    473.55     473.673   ]
  [473.796    473.919    474.042    ... 474.534    474.657    474.78    ]
  [474.903    475.026    475.149    ... 475.641    475.764    475.887   ]
  [476.01     476.133    476.256    ... 476.748    476.871    476.994   ]
  [477.117    477.24     477.363    ... 477.855    477.978    478.101   ]
  [478.224    478.347    478.47     ... 478.962    479.085    479.208   ]
  [479.331    479.454    479.577    ... 480.069    480.192    480.315   ]]

 [[480.438    480.561    480.684    ... 481.176    481.299    481.422   ]
  [481.545    481.668    481.791    ... 482.283    482.406    482.529   ]
  [482.652    482.775    482.898    ... 483.39     483.513    483.636   ]
  [483.759    483.882    484.005    ... 484.497    484.62     484.743   ]
  [484.866    484.989    485.112    ... 485.604    485.727    485.85    ]
  [485.973    486.096    486.219    ... 486.711    486.834    486.957   ]
  [487.08     487.203    487.326    ... 487.818    487.941    488.064   ]]

 [[488.187    488.31     488.433    ... 488.925    489.048    489.171   ]
  [489.294    489.417    489.54     ... 490.032    490.155    490.278   ]
  [490.401    490.524    490.647    ... 491.139    491.262    491.385   ]
  [491.508    491.631    491.754    ... 492.246    492.369    492.492   ]
  [492.615    492.738    492.861    ... 493.353    493.476    493.599   ]
  [493.722    493.845    493.968    ... 494.46     494.583    494.706   ]
  [494.829    494.952    495.075    ... 495.567    495.69     495.813   ]]])");
}

}  // namespace
}  // namespace mediapipe
