#include "mediapipe/framework/api2/stream/tensor_to_joints.h"

#include "mediapipe/calculators/tensor/tensor_to_joints_calculator.h"
#include "mediapipe/calculators/tensor/tensor_to_joints_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe::api2::builder {

namespace {}  // namespace

Stream<JointList> ConvertTensorToJointsAtIndex(Stream<Tensor> tensor,
                                               const int num_joints,
                                               const int start_index,
                                               Graph& graph) {
  auto& to_joints = graph.AddNode("TensorToJointsCalculator");
  auto& to_joints_options =
      to_joints.GetOptions<TensorToJointsCalculatorOptions>();
  to_joints_options.set_num_joints(num_joints);
  to_joints_options.set_start_index(start_index);
  tensor.ConnectTo(to_joints[TensorToJointsCalculator::kInTensor]);
  return to_joints[TensorToJointsCalculator::kOutJoints];
}

}  // namespace mediapipe::api2::builder
