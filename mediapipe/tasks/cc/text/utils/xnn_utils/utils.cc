#include "mediapipe/tasks/cc/text/utils/xnn_utils/utils.h"

namespace mediapipe {
namespace xnn_utils {

std::vector<float> FillXnnRoPEWeights(size_t max_seq_len, size_t num_channels) {
  std::vector<float> out_array(max_seq_len * num_channels);
  for (size_t ch_id = 0; ch_id < num_channels / 2; ++ch_id) {
    auto timescale = std::pow(1e-4, 2.0 * ch_id / num_channels);
    for (size_t seq_id = 0; seq_id < max_seq_len; ++seq_id) {
      auto sinusoid_inp = seq_id * timescale;
      out_array[seq_id * num_channels + ch_id] = cos(sinusoid_inp);
      out_array[seq_id * num_channels + ch_id + num_channels / 2] =
          sin(sinusoid_inp);
    }
  }
  return out_array;
}

}  // namespace xnn_utils
}  // namespace mediapipe
