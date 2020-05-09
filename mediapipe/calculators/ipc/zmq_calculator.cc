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

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_buffer.h"

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
        cc->Inputs().Get("FACE_LANDMARKS", 0).SetAny();
        cc->Inputs().Get("HAND_DETECTIONS", 0).SetAny();
        cc->Inputs().Get("IMAGE", 0).SetAny();
        // cc->SetInputStreamHandler("ImmediateInputStreamHandler");
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
            PublishJson("HandLandmarks", ConvertLandmarkListsToJson(landmarks));
        }

        if (!cc->Inputs().Tag("FACE_LANDMARKS").IsEmpty())
        {
            const auto &landmark_lists =
                cc->Inputs().Tag("FACE_LANDMARKS").Get<std::vector<NormalizedLandmarkList>>();

            PublishJson("FaceLandmarks", ConvertLandmarkListsToJson(landmark_lists));
        }

        if (!cc->Inputs().Tag("NORM_RECTS").IsEmpty())
        {
            const auto &norm_rects =
                cc->Inputs().Tag("NORM_RECTS").Get<std::vector<NormalizedRect>>();
            const auto &detections =
                cc->Inputs().Tag("HAND_DETECTIONS").Get<std::vector<Detection>>();
            const auto& image_frame = 
                cc->Inputs().Tag("IMAGE").Get<mediapipe::GpuBuffer>();
            const auto &landmark_lists =
                cc->Inputs().Tag("LANDMARKS").Get<std::vector<NormalizedLandmarkList>>();
            const auto& landmark_lists2 = ConvertLandmarkListsToJson(landmark_lists);
    
            assert(norm_rects.size() == detections.size());
            if (norm_rects.size() != landmark_lists2.size()) {
                LOG(INFO) << "BUG";
            }
            if (norm_rects.size() > 0)
            {
                json data = json({});
                data["hands"] = json::array();

                for (int i = 0; i < norm_rects.size(); i++)
                {
                    const auto& norm_rect = norm_rects[i];
                    const auto& detection = detections[i];

                    if (norm_rect.width() == 0.0 && norm_rect.height() == 0.0 && norm_rect.x_center() == 0.0 && norm_rect.y_center() == 0.0 && norm_rect.rect_id() == 0)
                    {
                        continue;
                    }
                    // const auto& landmarks = landmark_lists2[i]["landmarks"];
                    // LOG(INFO) << "Inside" << landmark_lists2;
                    json empty_object_explicit = json::object();
                    empty_object_explicit["width"] = norm_rect.width();
                    empty_object_explicit["height"] = norm_rect.height();
                    empty_object_explicit["x_center"] = norm_rect.x_center();
                    empty_object_explicit["y_center"] = norm_rect.y_center();
                    empty_object_explicit["rect_id"] = norm_rect.rect_id();
                    empty_object_explicit["image_width"] = image_frame.width();
                    empty_object_explicit["image_height"] = image_frame.height();
                    // LOG(INFO) << landmarks;
                    if (landmark_lists2.size() >= (i + 1)) {
                        // LOG(INFO) << (landmark_lists2.size() - 1);
                        // LOG(INFO) << i;
                        const auto& landmarks = landmark_lists2[i]["landmarks"];
                        empty_object_explicit["landmarks"] = landmarks;
                    }
                    // empty_object_explicit["track_id"] = norm_rect.id();
                    data["hands"].push_back(empty_object_explicit);
                }
                data["timestamp"] = cc->InputTimestamp().Microseconds();
                PublishJson("Detection", data);
            }
        }

        return ::mediapipe::OkStatus();
    }

    void PublishJson(const std::string &topic, const json &json_data)
    {
        std::string s = json_data.dump();
        // std::string topic = topic;

        zmq::message_t message(topic.size());
        memcpy(message.data(), topic.c_str(), topic.size());
        socket.send(message, ZMQ_SNDMORE);

        zmq::message_t message2(s.size());
        memcpy(message2.data(), s.c_str(), s.size());
        socket.send(message2);

        // std::cout << "Publishing" << s << std::endl;
    }

    json ConvertLandmarkListsToJson(const std::vector<NormalizedLandmarkList> &landmark_lists)
    {
        json landmark_list_json = json::array();
        for (const auto &landmark_list : landmark_lists)
        {
            json data = json({});
            data["landmarks"] = json::array();
            for (int i = 0; i < landmark_list.landmark_size(); ++i)
            {
                const NormalizedLandmark &landmark = landmark_list.landmark(i);
                json landmark_json = json::array();
                landmark_json.push_back(landmark.x());
                landmark_json.push_back(landmark.y());
                landmark_json.push_back(landmark.z());
                data["landmarks"].push_back(landmark_json);
            }
            landmark_list_json.push_back(data);
        }
        return landmark_list_json;
    }

private:
    zmq::context_t context{1};
    zmq::socket_t socket{context, ZMQ_PUB};
};
REGISTER_CALCULATOR(ZmqCalculator);

} // namespace mediapipe
