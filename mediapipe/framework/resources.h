#ifndef MEDIAPIPE_FRAMEWORK_RESOURCES_H_
#define MEDIAPIPE_FRAMEWORK_RESOURCES_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace mediapipe {

// Represents an interface to load resources in calculators and subgraphs.
//
// Should be accessed through `CalculatorContext::GetResources` and
// `SubgraphContext::GetResources`.
//
// Can be configured per graph by setting custom object through
// `kResourcesService` on `CalculatorGraph`.
class Resources {
 public:
  struct Options {
    bool read_as_binary = true;
  };

  virtual ~Resources() = default;

  // Gets resource contents by resource id.
  //
  // For backward compatibility with `GetResourceContents`, `resource_id` for
  // the default `Resources` implementation is currently a path and, depending
  // on the platform and other factors (like setting static AssetManager on
  // Android) other options are possible (e.g. returning a resource from Android
  // assets or loading from "content://..." URIs).
  //
  // NOTE: can be accessed simultaneously from multiple threads.
  virtual absl::Status ReadContents(absl::string_view resource_id,
                                    std::string& output,
                                    const Options& options) const = 0;

  // Gets resource contents by resource id.
  //
  // For backward compatibility with `GetResourceContents`, resource_id for
  // the default `Resources` implementation is currently a path and, depending
  // on the platform and other factors (like setting static AssetManager on
  // Android) other options are possible (e.g. returning a resource from Android
  // assets or loading from "content://..." URIs).
  //
  // NOTE: can be accessed simultaneously from multiple threads.
  inline absl::Status ReadContents(absl::string_view resource_id,
                                   std::string& output) const {
    return ReadContents(resource_id, output, Options());
  }
};

// `Resources` object which can be used in place of `GetResourceContents`.
std::unique_ptr<Resources> CreateDefaultResources();

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_RESOURCES_H_
