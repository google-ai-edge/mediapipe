#ifndef MEDIAPIPE_FRAMEWORK_RESOURCES_H_
#define MEDIAPIPE_FRAMEWORK_RESOURCES_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace mediapipe {

class Resource {
 public:
  Resource(const Resource&) = delete;
  Resource& operator=(const Resource&) = delete;
  virtual ~Resource() = default;

  const void* data() const { return data_; }
  size_t length() const { return length_; }

  // For use with APIs that prefer a string_view.
  absl::string_view ToStringView() const {
    return absl::string_view(static_cast<const char*>(data()), length());
  }

 protected:
  // For use by subclasses
  Resource(const void* data, size_t length) : data_(data), length_(length) {}

 private:
  const void* data_;
  size_t length_;
};

// Creates a resource which represents a string.
std::unique_ptr<Resource> MakeStringResource(std::string&& s);

// Creates a resource whose destructor does nothing.
//
// Useful when some higher level is responsible for allocation/deletion of the
// actual data blocks.
std::unique_ptr<Resource> MakeNoCleanupResource(const void* data,
                                                size_t length);

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
  // NOTE: Reasources::ReadContents may involve unnecessary memory copies, so
  // Resources::Get is preferable, except cases where `absl::string_view` cannot
  // be used (e.g. `istringstream`), but first consider alternatives like
  // `ForEachLine(absl::string_view, ...)`.
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
  // NOTE: Reasources::ReadContents may involve unnecessary memory copies, so
  // Resources::Get is preferable, except cases where `absl::string_view` cannot
  // be used (e.g. `istringstream`), but first consider alternatives like
  // `ForEachLine(absl::string_view, ...)`.
  //
  // NOTE: can be accessed simultaneously from multiple threads.
  inline absl::Status ReadContents(absl::string_view resource_id,
                                   std::string& output) const {
    return ReadContents(resource_id, output, Options());
  }

  // Gets a resource by resource id.
  //
  // For backward compatibility with `GetResourceContents`, resource_id for
  // the default `Resources` implementation is currently a path and, depending
  // on the platform and other factors (like setting static AssetManager on
  // Android) other options are possible (e.g. returning a resource from Android
  // assets or loading from "content://..." URIs).
  virtual absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const = 0;

  // Gets a resource contents by resource id.
  //
  // For backward compatibility with `GetResourceContents`, resource_id for
  // the default `Resources` implementation is currently a path and, depending
  // on the platform and other factors (like setting static AssetManager on
  // Android) other options are possible (e.g. returning a resource from Android
  // assets or loading from "content://..." URIs).
  inline absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id) const {
    return Get(resource_id, Options());
  }
};

// `Resources` object which can be used in place of `GetResourceContents`.
std::unique_ptr<Resources> CreateDefaultResources();

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_RESOURCES_H_
