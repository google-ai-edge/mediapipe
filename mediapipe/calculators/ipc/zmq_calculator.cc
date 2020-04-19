// Copyright 2019 The MediaPipe Authors.
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

#include <algorithm>
#include <utility>
#include <vector>
#include <time.h>

#include <zmq.hpp>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/header_util.h"

#include "json.hpp"

using json = nlohmann::json;

#define within(num) (int)((float)num * random() / (RAND_MAX + 1.0))

namespace mediapipe
{

class ZmqCalculator : public CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(CalculatorContract *cc)
    {
        cc->Inputs().Get("LANDMARKS", 0).SetAny();
        cc->Inputs().Get("NORM_RECTS", 0).SetAny();
        cc->SetInputStreamHandler("ImmediateInputStreamHandler");
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Open(CalculatorContext *cc) final
    {
        std::cout << "opened" << std::endl;
        socket.bind("tcp://*:5555");
        return ::mediapipe::OkStatus();
    }

    bool Allow() { return true; }

    ::mediapipe::Status Process(CalculatorContext *cc) final
    {
        if (!cc->Inputs().Tag("LANDMARKS").IsEmpty())
        {
            const auto &landmarks =
                cc->Inputs().Tag("LANDMARKS").Get<std::vector<NormalizedLandmarkList>>();
        }

        if (!cc->Inputs().Tag("NORM_RECTS").IsEmpty())
        {
            const auto &norm_rects =
                cc->Inputs().Tag("NORM_RECTS").Get<std::vector<NormalizedRect>>();
            if (norm_rects.size() > 0)
            {
                json data = json({});
                data["hands"] = json::array();

                for (const auto &norm_rect : norm_rects)
                {
                    if (norm_rect.width() == 0.0 && norm_rect.height() == 0.0 && norm_rect.x_center() == 0.0 && norm_rect.y_center() == 0.0 && norm_rect.rect_id() == 0) {
                        continue;
                    }
                    json empty_object_explicit = json::object();
                    empty_object_explicit["width"] = norm_rect.width();
                    empty_object_explicit["height"] = norm_rect.height();
                    empty_object_explicit["x_center"] = norm_rect.x_center();
                    empty_object_explicit["y_center"] = norm_rect.y_center();
                    empty_object_explicit["rect_id"] = norm_rect.rect_id();
                    data["hands"].push_back(empty_object_explicit);
                }
                std::string s = data.dump();
                std::string topic = "Detection";

                zmq::message_t message(topic.size());
                memcpy(message.data(), topic.c_str(), topic.size());
                socket.send(message, ZMQ_SNDMORE);

                zmq::message_t message2(s.size());
                memcpy(message2.data(), s.c_str(), s.size());
                socket.send(message2);
                
                std::cout << "Publishing" << s << std::endl;
            }
        }

        return ::mediapipe::OkStatus();
    }

private:
    zmq::context_t context{1};
    zmq::socket_t socket{context, ZMQ_PUB};
};
REGISTER_CALCULATOR(ZmqCalculator);

} // namespace mediapipe
