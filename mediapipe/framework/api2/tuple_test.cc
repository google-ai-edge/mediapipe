#include "mediapipe/framework/api2/tuple.h"

#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace api2 {
namespace internal {
namespace {

template <typename A, typename B>
constexpr bool same_type(A, B) {
  return false;
}

template <typename A>
constexpr bool same_type(A, A) {
  return true;
}

template <std::size_t... I>
using iseq = std::index_sequence<I...>;

TEST(TupleTest, IndexSeq) {
  EXPECT_TRUE(
      same_type(iseq<0, 1, 2>(), index_sequence_cat(iseq<0, 1>(), iseq<2>())));
  EXPECT_TRUE(same_type(iseq<0, 1, 2>(),
                        index_sequence_cat(iseq<0, 1>(), iseq<>(), iseq<2>())));
}

TEST(TupleTest, FilteredIndices) {
  EXPECT_TRUE(same_type(
      filtered_tuple_indices<std::is_integral>(std::tuple<int, float, char>()),
      iseq<0, 2>()));
}

TEST(TupleTest, SelectIndices) {
  auto t = std::make_tuple(5.0, 10, "hi");
  EXPECT_EQ((select_tuple_indices(t, iseq<0, 2>())),
            (std::make_tuple(5.0, "hi")));
}

TEST(TupleTest, FilterTuple) {
  auto t = std::make_tuple(5.0, 10, "hi");
  EXPECT_EQ((filter_tuple<std::is_integral>(t)), (std::make_tuple(10)));
}

TEST(TupleTest, FilterTupleRefs) {
  auto t = std::make_tuple(5.0, 10, "hi");
  auto tr = filter_tuple<std::is_integral>(t);
  int x;
  EXPECT_TRUE(same_type(tr, std::tuple<int&>{x}));
  EXPECT_FALSE(same_type(tr, std::tuple<int>{x}));
  auto tr_copy =
      std::apply([](auto&&... item) { return std::make_tuple(item...); },
                 filter_tuple<std::is_integral>(t));
  EXPECT_TRUE(same_type(tr_copy, std::tuple<int>{x}));
}

struct is_integral {
  template <class W>
  constexpr bool operator()(W&&) {
    return std::is_integral<typename W::type>{};
  }
};

TEST(TupleTest, FilteredIndices2) {
  EXPECT_TRUE(same_type(
      filtered_tuple_indices<is_integral>(std::tuple<int, float, char>()),
      iseq<0, 2>()));
}

// TEST(TupleTest, FilterTuple2) {
//   auto t = std::make_tuple(5.0, 10, "hi");
//   auto is_int = [](auto&& x) {
//     return std::is_integral_v<decltype(x)>;
//   };
//   EXPECT_EQ((filter_tuple(is_int, t)), (std::make_tuple(10)));
// }

TEST(TupleTest, ForEach) {
  auto t = std::make_tuple(5.0, 10, "hi");
  std::vector<std::string> s;
  tuple_for_each([&s](auto&& item) { s.push_back(absl::StrCat(item)); }, t);
  EXPECT_EQ(s, (std::vector<std::string>{"5", "10", "hi"}));
}

TEST(TupleTest, ForEachWithIndex) {
  auto t = std::make_tuple(5.0, 10, "hi");
  std::vector<std::string> s;
  tuple_for_each(
      [&s](auto&& item, std::size_t i) {
        s.push_back(absl::StrCat(i, ":", item));
      },
      t);
  EXPECT_EQ(s, (std::vector<std::string>{"0:5", "1:10", "2:hi"}));
}

TEST(TupleTest, ForEachZip) {
  auto t = std::make_tuple(5.0, 10, "hi");
  auto u = std::make_tuple(2.0, 3, "lo");
  std::vector<std::string> s;
  tuple_for_each(
      [&s, &u](auto&& item, auto i_const) {
        constexpr std::size_t i = decltype(i_const)::value;
        s.push_back(absl::StrCat(i, ":", item, ",", std::get<i>(u)));
      },
      t);
  EXPECT_EQ(s, (std::vector<std::string>{"0:5,2", "1:10,3", "2:hi,lo"}));
}

TEST(TupleTest, Apply) {
  auto t = std::make_tuple(5.0, 10, "hi");
  std::string s = tuple_apply(
      [](float f, int i, const char* s) { return absl::StrCat(f, i, s); }, t);
  EXPECT_EQ(s, "510hi");
}

TEST(TupleTest, Map) {
  auto t = std::make_tuple(5.0, 10, 2L);
  auto t2 = map_tuple([](auto x) { return x * 2; }, t);
  EXPECT_EQ(t2, std::make_tuple(10.0, 20, 4L));
}

TEST(TupleFind, Find) {
  auto t = std::make_tuple(5.0, 10, 2L);
  auto i = tuple_find([](auto x) { return x > 3; }, t);
  EXPECT_EQ(i, 0);
}

TEST(TupleFind, Flatten) {
  auto t1 = std::make_tuple(5.0, 10);
  auto t2 = std::make_tuple(2L);
  auto t = std::make_tuple(t1, t2);
  auto tf = flatten_tuple(t);
  EXPECT_EQ(tf, std::make_tuple(5.0, 10, 2L));
}

}  // namespace
}  // namespace internal
}  // namespace api2
}  // namespace mediapipe
