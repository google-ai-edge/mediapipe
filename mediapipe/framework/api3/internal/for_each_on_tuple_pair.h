#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_FOR_EACH_ON_TUPLE_PAIR_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_FOR_EACH_ON_TUPLE_PAIR_H_

#include <cstddef>
#include <tuple>
#include <utility>

namespace mediapipe::api3 {

template <typename TupleA, typename TupleB, typename Func, size_t... I>
void ForEachOnTuplePairImpl(const TupleA& ta, const TupleB& tb, Func f,
                            std::index_sequence<I...>) {
  (f(std::get<I>(ta), std::get<I>(tb)), ...);
}

// Iterates over two tuples (of same size) and invokes `f` for pair of tuples`
// elements at the same index:
// - f(tuple_a_el_at_0, tuple_b_el_at0)
// - ...
// - f(tuple_a_el_at_N, tuple_b_el_atN)
template <typename TupleA, typename TupleB, typename F>
void ForEachOnTuplePair(const TupleA& ta, const TupleB& tb, F f) {
  static_assert(std::tuple_size_v<TupleA> == std::tuple_size_v<TupleB>);
  return ForEachOnTuplePairImpl(
      ta, tb, f, std::make_index_sequence<std::tuple_size_v<TupleA>>{});
}

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_FOR_EACH_ON_TUPLE_PAIR_H_
