//*****************************************************************************
// Copyright 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <ctime>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <openvino/openvino.hpp>

#include "mediapipe/calculators/ovms/openvinoinferencedumputils.h"

namespace mediapipe {
    
std::unordered_map<std::string, int> dumpCounters;

using InferenceInput = std::map<std::string, ov::Tensor>;

#define TYPE_CASE(ENUM, TYPE)                                                       \
    case ENUM: {                                                                    \
        const TYPE* value = reinterpret_cast<TYPE*>(tensor.data());   \
        dumpStream << " Tensor: [ ";                                                \
        for (int x = 0; x < tensor.get_size(); x++) {                               \
            dumpStream << value[x] << " ";                            \
        }                                                                           \
        dumpStream << " ]";                                                         \
        break;                                                                      \
    }                                                                               \

static std::stringstream dumpOvTensor(const ov::Tensor& tensor) {
    std::stringstream dumpStream;
    switch (tensor.get_element_type()) {
        TYPE_CASE(ov::element::Type_t::f64, _Float64)
        TYPE_CASE(ov::element::Type_t::f32, _Float32)
        TYPE_CASE(ov::element::Type_t::i64, int64_t)
        TYPE_CASE(ov::element::Type_t::i32, int32_t)
        TYPE_CASE(ov::element::Type_t::i16, int16_t)
        TYPE_CASE(ov::element::Type_t::i8, int8_t)
        TYPE_CASE(ov::element::Type_t::u32, uint32_t)
        TYPE_CASE(ov::element::Type_t::u16, uint16_t)
        TYPE_CASE(ov::element::Type_t::u8, uint8_t)
        TYPE_CASE(ov::element::Type_t::boolean, bool)
        case ov::element::Type_t::bf16:
        case ov::element::Type_t::undefined:
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::f16:
        case ov::element::Type_t::i4:
        case ov::element::Type_t::u4:
        case ov::element::Type_t::u1: {
            dumpStream << " unsupported dump type: [ " << tensor.get_element_type() << " ]";
            break;
        }
    }

    return dumpStream;    
}

static bool isAbsolutePath(const std::string& path) {
        return !path.empty() && (path[0] == '/');
}

static std::string joinPath(std::initializer_list<std::string> segments) {
    std::string joined;

    for (const auto& seg : segments) {
        if (joined.empty()) {
            joined = seg;
        } else if (isAbsolutePath(seg)) {
            if (joined[joined.size() - 1] == '/') {
                joined.append(seg.substr(1));
            } else {
                joined.append(seg);
            }
        } else {
            if (joined[joined.size() - 1] != '/') {
                joined.append("/");
            }
            joined.append(seg);
        }
    }

    return joined;
}

static void writeToFile(std::stringstream& stream, std::string name)
{
    std::ofstream ofs;
    ofs.open(name);
    ofs << stream.rdbuf();
    ofs.close();

    return;
}

static std::string getTimestampString() {
    auto rawtime = std::make_unique<time_t>();
    time(rawtime.get());
    struct tm* timeinfo = localtime(rawtime.get());
    auto start = std::chrono::system_clock::now();
    std::stringstream timestampStream;
    timestampStream << timeinfo->tm_year << "_" << timeinfo->tm_mon << "_" << timeinfo->tm_mday << "_" ;
    timestampStream << timeinfo->tm_hour << "_" << timeinfo->tm_min << "_" << timeinfo->tm_sec << "_";
    using namespace std::chrono;
    timestampStream << duration_cast<milliseconds>(start.time_since_epoch()).count();
    return timestampStream.str();
}

static int getAndIncerementCounter(const std::string& name) {
    if (dumpCounters.find(name) == dumpCounters.end())
        dumpCounters[name] = 0;

    return dumpCounters[name]++;
}

static const std::string TIMESTAMP_STRING = getTimestampString();

void dumpOvTensorInput(const InferenceInput& input, const std::string& dumpDirectoryName) {
    std::stringstream dumpStream;
    std::string fname = std::string("./dump");
    fname = joinPath({fname,TIMESTAMP_STRING});
    std::filesystem::create_directories(fname);
    fname = joinPath({fname, dumpDirectoryName + std::to_string(getAndIncerementCounter(dumpDirectoryName))});
    for (const auto& [name, inputTensor] : input) {
        dumpStream << " Name: " << name;
        dumpStream << " Shape: " << inputTensor.get_shape();
        dumpStream << " Type: " << inputTensor.get_element_type();
        dumpStream << " Byte size: " << inputTensor.get_byte_size();
        dumpStream << " Size: " << inputTensor.get_size();
        dumpStream << dumpOvTensor(inputTensor).str();
    }

    std::cout << "Dump filename: " << fname <<std::endl;
    writeToFile(dumpStream, fname);
}

}  // namespace mediapipe
