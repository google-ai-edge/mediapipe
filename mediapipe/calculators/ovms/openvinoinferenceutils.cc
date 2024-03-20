//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ovms.h"           // NOLINT
namespace mediapipe {

// Function from ovms/src/string_utils.h
bool startsWith(const std::string& str, const std::string& prefix) {
    auto it = prefix.begin();
    bool sizeCheck = (str.size() >= prefix.size());
    if (!sizeCheck) {
        return false;
    }
    bool allOf = std::all_of(str.begin(),
        std::next(str.begin(), prefix.size()),
        [&it](const char& c) {
            return c == *(it++);
        });
    return allOf;
}

// Function from ovms/src/string_utils.h
std::vector<std::string> tokenize(const std::string& str, const char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(str);
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

// Function from ovms/src/string_utils.h
bool endsWith(const std::string& str, const std::string& match) {
    auto it = match.begin();
    return str.size() >= match.size() &&
           std::all_of(std::next(str.begin(), str.size() - match.size()), str.end(), [&it](const char& c) {
               return ::tolower(c) == ::tolower(*(it++));
           });
}

OVMS_LogLevel StringToLogLevel(const std::string& logLevel){
    if (logLevel == "3")
        return OVMS_LOG_ERROR;
    if (logLevel == "1")
        return OVMS_LOG_DEBUG;
    if (logLevel == "0")
        return OVMS_LOG_TRACE;
    if (logLevel == "2")
        return OVMS_LOG_INFO;

    return OVMS_LOG_INFO;
}

std::string LogLevelToString(OVMS_LogLevel log_level) {
    switch (log_level) {
    case OVMS_LOG_INFO:
        return "INFO";
    case OVMS_LOG_ERROR:
        return "ERROR";
    case OVMS_LOG_DEBUG:
        return "DEBUG";
    case OVMS_LOG_TRACE:
        return "TRACE";
    case OVMS_LOG_WARNING:
        return "WARNING";
        
    }

    return "unsupported";
}

}  // namespace mediapipe
