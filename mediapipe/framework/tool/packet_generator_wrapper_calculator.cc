#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_registry.h"
#include "mediapipe/framework/output_side_packet.h"
#include "mediapipe/framework/packet_generator.h"
#include "mediapipe/framework/tool/packet_generator_wrapper_calculator.pb.h"

namespace mediapipe {

class PacketGeneratorWrapperCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    const auto& options =
        cc->Options<::mediapipe::PacketGeneratorWrapperCalculatorOptions>();
    ASSIGN_OR_RETURN(auto static_access,
                     mediapipe::internal::StaticAccessToGeneratorRegistry::
                         CreateByNameInNamespace(options.package(),
                                                 options.packet_generator()));
    MP_RETURN_IF_ERROR(static_access->FillExpectations(
                           options.options(), &cc->InputSidePackets(),
                           &cc->OutputSidePackets()))
            .SetPrepend()
        << options.packet_generator() << "::FillExpectations() failed: ";
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<::mediapipe::PacketGeneratorWrapperCalculatorOptions>();
    ASSIGN_OR_RETURN(auto static_access,
                     mediapipe::internal::StaticAccessToGeneratorRegistry::
                         CreateByNameInNamespace(options.package(),
                                                 options.packet_generator()));
    mediapipe::PacketSet output_packets(cc->OutputSidePackets().TagMap());
    MP_RETURN_IF_ERROR(static_access->Generate(options.options(),
                                               cc->InputSidePackets(),
                                               &output_packets))
            .SetPrepend()
        << options.packet_generator() << "::Generate() failed: ";
    for (auto id = output_packets.BeginId(); id < output_packets.EndId();
         ++id) {
      cc->OutputSidePackets().Get(id).Set(output_packets.Get(id));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(PacketGeneratorWrapperCalculator);

}  // namespace mediapipe
