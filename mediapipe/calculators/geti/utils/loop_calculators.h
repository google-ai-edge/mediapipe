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
#ifndef LOOP_CALCULATORS_H
#define LOOP_CALCULATORS_H

#include <models/input_data.h>
#include <models/results.h>

#include <vector>

#include "../inference/geti_calculator_base.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"
#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/framework/calculator_framework.h"
#include "../utils/data_structures.h"

namespace mediapipe {

using BeginLoopRectanglePredictionCalculator =
    BeginLoopCalculator<std::vector<geti::RectanglePrediction>>;
using EndLoopRectanglePredictionsCalculator =
    EndLoopCalculator<std::vector<geti::RectanglePrediction>>;
using EndLoopPolygonPredictionsCalculator =
    EndLoopCalculator<std::vector<std::vector<geti::PolygonPrediction>>>;

using BeginLoopModelApiDetectionCalculator =
    BeginLoopCalculator<std::vector<geti::RectanglePrediction>>;
using EndLoopModelApiDetectionClassificationCalculator =
    EndLoopCalculator<std::vector<geti::RectanglePrediction>>;
using EndLoopModelApiDetectionSegmentationCalculator =
    EndLoopCalculator<std::vector<std::vector<geti::PolygonPrediction>>>;
}  // namespace mediapipe

#endif  // LOOP_CALCULATORS_H
