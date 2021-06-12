#include "mediapipe/framework/api2/type_list.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace api2 {
namespace types {
namespace {

template <typename A, typename B>
constexpr bool same_type(A, B) {
  return false;
}

template <typename A>
constexpr bool same_type(A, A) {
  return true;
}

struct Foo {};
struct Bar {};
struct Baz {};

TEST(TypeListFTest, SameType) {
  EXPECT_FALSE(same_type(List<Foo>{}, List<>{}));
  EXPECT_TRUE(same_type(List<Foo>{}, List<Foo>{}));
}

TEST(TypeListFTest, Length) {
  EXPECT_EQ(length(List<float, int>{}), 2);
  EXPECT_EQ(length(List<>{}), 0);
}

TEST(TypeListFTest, Head) {
  using Empty = List<>;
  using ListA = List<Foo, Bar>;
  EXPECT_TRUE(same_type(Wrap<Foo>{}, head(ListA{})));
  EXPECT_TRUE(same_type(Wrap<void>{}, head(Empty{})));
}

TEST(TypeListFTest, Concat) {
  using Empty = List<>;
  using ListA = List<Foo>;

  EXPECT_TRUE(same_type(ListA{}, concat(ListA{}, Empty{})));
  EXPECT_TRUE(same_type(concat(ListA{}, Empty{}), ListA{}));

  using ListB = List<Bar, Baz>;
  EXPECT_TRUE(same_type(concat(ListA{}, ListB{}), List<Foo, Bar, Baz>{}));
}

TEST(TypeListFTest, Filter) {
  EXPECT_TRUE(same_type(filter<std::is_integral>(List<>{}), List<>{}));
  EXPECT_TRUE(same_type(filter<std::is_integral>(List<int, float, char>{}),
                        List<int, char>{}));
}

TEST(TypeListFTest, Filter2) {
  constexpr auto is_integral = [](auto x) {
    return std::is_integral<decltype(x)>{};
  };
  auto x = filter(is_integral, List<>{});
  EXPECT_TRUE(same_type(x, List<>{}));
  auto y = filter(is_integral, List<int, float, char>{});
  EXPECT_TRUE(same_type(y, List<int, char>{}));
  auto z = filter([](auto x) { return std::is_integral<decltype(x)>{}; },
                  List<int, double>{});
  EXPECT_TRUE(same_type(z, List<int>{}));
}

TEST(TypeListFTest, Find) {
  EXPECT_TRUE(same_type(find<std::is_integral>(List<>{}), Wrap<void>{}));
  EXPECT_TRUE(
      same_type(find<std::is_integral>(List<float, int>{}), Wrap<int>()));
}

TEST(TypeListFTest, Find2) {
  constexpr auto is_integral = [](auto x) {
    return std::is_integral<decltype(x)>{};
  };
  EXPECT_TRUE(same_type(find(is_integral, List<>{}), Wrap<void>{}));
  EXPECT_TRUE(same_type(find(is_integral, List<float, int>{}), Wrap<int>()));
}

TEST(TypeListFTest, Map) {
  EXPECT_TRUE(
      same_type(map<std::remove_cv>(List<const int, const float, const char>{}),
                List<int, float, char>{}));
}

TEST(TypeListFTest, Enumerate) {
  EXPECT_TRUE(same_type(enumerate(List<int, float, char>{}),
                        List<IndexedType<0, int>, IndexedType<1, float>,
                             IndexedType<2, char>>{}));
}

}  // namespace
}  // namespace types
}  // namespace api2
}  // namespace mediapipe
