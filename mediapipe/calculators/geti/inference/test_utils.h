/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */

#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "nlohmann/json.hpp"

using namespace nlohmann;

namespace geti {
static inline void RunGraph(
    mediapipe::Packet packet, mediapipe::CalculatorGraphConfig& graph_config,
    std::vector<mediapipe::Packet>& output_packets,
    std::map<std::string, mediapipe::Packet> inputSidePackets = {}) {
  mediapipe::tool::AddVectorSink("output", &graph_config, &output_packets);

  mediapipe::CalculatorGraph graph(graph_config);

  MP_ASSERT_OK(graph.StartRun(inputSidePackets));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", packet.At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
}

static inline bool json_equals(json::const_reference source,
                               json::const_reference target, float epsilon) {
  if (source == target) {
    return true;
  }

  if (source.type() != target.type()) {
    return false;
  }

  switch (source.type()) {
    case json::value_t::array:
      if (source.size() != target.size()) {
        return false;
      }
      for (size_t i = 0; i < source.size(); i++) {
        if (!json_equals(source[i], target[i], epsilon)) {
          return false;
        }
      }
      return true;
    case json::value_t::object:
      for (auto it = source.cbegin(); it != source.cend(); ++it) {
        if (target.find(it.key()) == target.end() ||
            !json_equals(it.value(), target[it.key()], epsilon)) {
          return false;
        }
      }
      for (auto it = target.cbegin(); it != target.cend(); ++it) {
        if (source.find(it.key()) == source.end() ||
            !json_equals(it.value(), source[it.key()], epsilon)) {
          return false;
        }
      }
      return true;
    case json::value_t::number_float:
      return std::abs((float)source - (float)target) <= epsilon;
    case json::value_t::null:
    case json::value_t::string:
    case json::value_t::boolean:
    case json::value_t::number_integer:
    case json::value_t::number_unsigned:
    // case json::value_t::binary:
    case json::value_t::discarded:
    default:
      return false;
  }
}

}  // namespace geti

#endif  // TEST_UTILS_H_
