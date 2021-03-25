#ifndef MEDIAPIPE_UTIL_RESOURCE_UTIL_CUSTOM_H_
#define MEDIAPIPE_UTIL_RESOURCE_UTIL_CUSTOM_H_

#include <string>

#include "mediapipe/framework/port/status.h"

namespace mediapipe {

typedef std::function<absl::Status(const std::string&, std::string*)>
    ResourceProviderFn;

// Overrides the behavior of GetResourceContents.
void SetCustomGlobalResourceProvider(ResourceProviderFn fn);

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_RESOURCE_UTIL_CUSTOM_H_
