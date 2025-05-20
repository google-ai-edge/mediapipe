#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_HAS_UPDATE_CONTRACT_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_HAS_UPDATE_CONTRACT_H_

#include <type_traits>

namespace mediapipe::api3 {

template <typename I, typename C, typename = std::void_t<>>
struct HasUpdateContractHelper : std::false_type {};

template <typename I, typename C>
struct HasUpdateContractHelper<
    I, C, std::void_t<decltype(I::UpdateContract(std::declval<C&>()))>>
    : std::true_type {};

template <typename I, typename C>
static constexpr bool kHasUpdateContract = HasUpdateContractHelper<I, C>{};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_HAS_UPDATE_CONTRACT_H_
