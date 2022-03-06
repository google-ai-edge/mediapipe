// Copyright 2019 Prince Patel
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include <vector>


// This calculator supports a input stream with Color values of Red, Green, and Blue
// TODO: Remove this requirement by replacing the typed input streams
//
//	input_stream: "Red"
//	input_stream: "Green"
//	input_stream: "Green"
//
// Output streams:
//   RGB_OUT:      The output video strem with array of values.
//


namespace mediapipe {

class ColorSliderCalculator : public CalculatorBase {
  public:
	  ColorSliderCalculator() = default;
	  ~ColorSliderCalculator() override = default;

	  static absl::Status GetContract(CalculatorContract* cc);
	  absl::Status Open(CalculatorContext* cc) override;
	  absl::Status Process(CalculatorContext* cc) override;
	  void make_array(int r,int g,int b,std::array<int,3>* out);
};
REGISTER_CALCULATOR(ColorSliderCalculator);

//static
absl::Status ColorSliderCalculator::GetContract (CalculatorContract *cc){
    cc->Inputs().Index(0).Set<int>();
    cc->Inputs().Index(1).Set<int>();
    cc->Inputs().Index(2).Set<int>();

    if (cc->Outputs().HasTag("RGB_OUT")){
      cc->Outputs().Tag("RGB_OUT").Set<std::array<int,3>>();
    }
    return absl::OkStatus();
}

absl::Status ColorSliderCalculator::Open(CalculatorContext* cc) {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
}

absl::Status ColorSliderCalculator::Process(CalculatorContext* cc) {
    if (cc->Inputs().NumEntries() == 0) {
          return tool::StatusStop();
        }
    int red_buffer = cc->Inputs().Index(0).Value().Get<int>();
    int green_buffer = cc->Inputs().Index(1).Value().Get<int>();
    int blue_buffer = cc->Inputs().Index(2).Value().Get<int>();
    auto out = absl::make_unique<std::array<int,3>>();
    make_array(red_buffer,green_buffer,blue_buffer, out.get());

    cc->Outputs().Tag("RGB_OUT").Add(out.release(), cc->InputTimestamp());
    LOG(INFO) << "Color Slider Calculator Runner" << red_buffer << " " << green_buffer << " " << blue_buffer << "\n";
    return absl::OkStatus();
}

void ColorSliderCalculator::make_array(int r,int g,int b,std::array<int,3>* out){
    (*out)[0] =r;
    (*out)[1] =g;
    (*out)[2] =b;
}

} // namespace mediapipe
