#include "mediapipe/util/resources_test_util.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"            // from @com_google_absl
#include "absl/status/statusor.h"          // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/strings/string_view.h"      // from @com_google_absl
#include "mediapipe/framework/resources.h"

namespace mediapipe {
namespace {

class InMemoryResources : public Resources {
 public:
  explicit InMemoryResources(
      absl::flat_hash_map<std::string, std::string> resources)
      : resources_(std::move(resources)) {}

  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const final {
    auto it = resources_.find(resource_id);
    if (it != resources_.end()) {
      return MakeNoCleanupResource(it->second.data(), it->second.size());
    }
    return absl::NotFoundError(absl::StrCat(resource_id, " not found."));
  }

 private:
  absl::flat_hash_map<std::string, std::string> resources_;
};

}  // namespace

std::unique_ptr<Resources> CreateInMemoryResources(
    absl::flat_hash_map<std::string, std::string> resources) {
  return std::make_unique<InMemoryResources>(std::move(resources));
}

}  // namespace mediapipe
