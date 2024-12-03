#ifndef MEDIAPIPE_FRAMEWORK_RESOURCES_H_
#define MEDIAPIPE_FRAMEWORK_RESOURCES_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
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

  // Clients should strive to use `ToStringView` instead wherever possible.
  //
  // If `absl::string_view` doesn't work for some reason, this function can be
  // used to get underlying `std::string` if resource has one (e.g. resource
  // constructed with `MakeStringResource`), or copy to a new `std::string`
  // (e.g. embedded data).
  //
  // Example:
  //   std::unique_ptr<Resource> resource = ...;
  //   std::string data = std::move(*resource).ReleaseOrCopyAsString();
  //
  virtual std::string ReleaseOrCopyAsString() && {
    return std::string(static_cast<const char*>(data()), length());
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

// Creates `Resources` object which enables resource mapping within a graph and
// can be used in place of `GetResourceContents`.
//
// `mapping` keys are resources ids.
//
// Example:
//
// `CalculatorGraphConfig`:
//   node {
//     ...
//     options {
//       [type.googleapis.com/...] {
//         model_path: "$MODEL"
//       }
//     }
//   }
//
// `CalculatorGraph` setup:
//
//   CalculatorGraph graph;
//   std::shared_ptr<Resources> resources = CreateDefaultResourcesWithMapping(
//       {{"$MODEL", "real/path/to/the/model"}});
//   graph.SetServiceObject(kResourcesService, std::move(resources));
//   graph.Initialize(std::move(config));
//
// As a result, when loading using ...Context::GetResources, not will be able
// to load the model from "real/path/to/the/model".
std::unique_ptr<Resources> CreateDefaultResourcesWithMapping(
    absl::flat_hash_map<std::string, std::string> mapping);

// Wraps `resources` to provide resources by resource id using a mapping when
// available.
//
// `mapping` keys are resources ids.
std::unique_ptr<Resources> CreateResourcesWithMapping(
    std::unique_ptr<Resources> resources,
    absl::flat_hash_map<std::string, std::string> mapping);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_RESOURCES_H_
