#include "mediapipe/framework/resources.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

namespace {

class DefaultResources : public Resources {
 public:
  absl::Status ReadContents(absl::string_view resource_id, std::string& output,
                            const Options& options) const final {
    return GetResourceContents(std::string(resource_id), &output,
                               options.read_as_binary);
  }
};

}  // namespace

std::unique_ptr<Resources> CreateDefaultResources() {
  return std::make_unique<DefaultResources>();
}

}  // namespace mediapipe
