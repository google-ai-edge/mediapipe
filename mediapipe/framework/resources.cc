#include "mediapipe/framework/resources.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/mlock_helpers.h"
#include "mediapipe/framework/deps/mmapped_file.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

namespace {

class StringResource : public Resource {
 public:
  explicit StringResource(std::unique_ptr<std::string> s)
      : Resource(s->data(), s->size()), s_(std::move(s)) {};

  std::string ReleaseOrCopyAsString() && final { return *std::move(s_); }

 private:
  std::unique_ptr<std::string> s_;
};

// A Resource whose destructor does nothing.  Useful when some higher level is
// responsible for allocation/deletion of the actual data blocks.
class NoCleanupResource : public Resource {
 public:
  NoCleanupResource(const void* data, size_t length) : Resource(data, length) {}
};

class MMapResource : public Resource {
 public:
  MMapResource(std::unique_ptr<file::MemoryMappedFile> mmapped_file,
               bool mlocked)
      : Resource(mmapped_file->BaseAddress(), mmapped_file->Length()),
        mmapped_file_(std::move(mmapped_file)),
        mlocked_(mlocked) {}

  absl::StatusOr<int> TryGetFd() const override {
    return mmapped_file_->TryGetFd();
  }

  ~MMapResource() override {
    if (mlocked_) {
      auto status =
          UnlockMemory(mmapped_file_->BaseAddress(), mmapped_file_->Length());
      if (!status.ok()) {
        ABSL_LOG(DFATAL) << status;
      }
    }
    auto status = mmapped_file_->Close();
    if (!status.ok()) {
      ABSL_LOG(DFATAL) << status;
    }
  }

 private:
  std::unique_ptr<file::MemoryMappedFile> mmapped_file_;
  bool mlocked_;
};

class DefaultResources : public Resources {
 public:
  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const final {
    const std::string path(resource_id);
    if (options.mmap_mode.has_value()) {
      const MMapMode mode = options.mmap_mode.value();
      // Try to resolve `resource_id` into a path.
      const absl::StatusOr<std::string> resolved_path =
          PathToResourceAsFile(path, /*shadow_copy=*/false);
      if (resolved_path.ok()) {
        auto status_or_mmap =
            MakeMMapResource(path,
                             /*mlock=*/mode == MMapMode::kMMapAndMLock);
        if (status_or_mmap.ok() || mode != MMapMode::kMMapOrRead) {
          return status_or_mmap;
        }
      } else if (mode != MMapMode::kMMapOrRead) {
        return resolved_path.status();
      }
    }

    // Try to load the resource as is.
    std::string output;
    const absl::Status status =
        GetResourceContents(path, &output, options.read_as_binary);
    if (status.ok()) {
      return MakeStringResource(std::move(output));
    }

    // Try the path resolution again, this time possibly with shadow copying.
    const absl::StatusOr<std::string> resolved_path_maybe_shadow =
        PathToResourceAsFile(path, /*shadow_copy=*/true);
    if (!resolved_path_maybe_shadow.ok()) {
      return tool::CombinedStatus(
          absl::StrCat("Failed to load resource: ", resource_id),
          {status, resolved_path_maybe_shadow.status()});
    }

    // Try to load by resolved path.
    absl::Status status_for_resolved = GetResourceContents(
        resolved_path_maybe_shadow.value(), &output, options.read_as_binary);
    if (status_for_resolved.ok()) {
      return MakeStringResource(std::move(output));
    }
    return tool::CombinedStatus(
        absl::StrCat("Failed to load resource: ", resource_id),
        {status, status_for_resolved});
  }
};

class ResourcesWithMapping : public Resources {
 public:
  explicit ResourcesWithMapping(
      std::unique_ptr<Resources> resources,
      absl::flat_hash_map<std::string, std::string> mapping)
      : resources_(std::move(resources)), mapping_(std::move(mapping)) {}

  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const final {
    return resources_->Get(MaybeIdFromMapping(resource_id), options);
  }

  absl::StatusOr<std::string> ResolveId(absl::string_view resource_id,
                                        const Options& options) const final {
    return resources_->ResolveId(MaybeIdFromMapping(resource_id), options);
  }

 private:
  absl::string_view MaybeIdFromMapping(absl::string_view resource_id) const {
    auto iter = mapping_.find(resource_id);
    absl::string_view resolved_res_id;
    if (iter != mapping_.end()) {
      resolved_res_id = iter->second;
    } else {
      resolved_res_id = resource_id;
    }
    return resolved_res_id;
  }

  std::unique_ptr<Resources> resources_;
  absl::flat_hash_map<std::string, std::string> mapping_;
};

}  // namespace

std::unique_ptr<Resource> MakeStringResource(std::string&& s) {
  return std::make_unique<StringResource>(
      std::make_unique<std::string>(std::move(s)));
}

std::unique_ptr<Resource> MakeNoCleanupResource(const void* data,
                                                size_t length) {
  return std::make_unique<NoCleanupResource>(data, length);
}

absl::StatusOr<std::unique_ptr<Resource>> MakeMMapResource(
    absl::string_view path, bool mlock) {
  auto mmap_or_error = file::MMapFile(path);
  if (!mmap_or_error.ok()) {
    return mmap_or_error.status();
  }
  std::unique_ptr<file::MemoryMappedFile> mmap = std::move(*mmap_or_error);

  if (mlock) {
    auto status = LockMemory(mmap->BaseAddress(), mmap->Length());
    if (!status.ok()) {
      return absl::UnavailableError(absl::StrCat("Locking memory for file '",
                                                 path, "' failed: ", status));
    }
  }
  return std::make_unique<MMapResource>(std::move(mmap), mlock);
}

std::unique_ptr<Resources> CreateDefaultResources() {
  return std::make_unique<DefaultResources>();
}

std::unique_ptr<Resources> CreateDefaultResourcesWithMapping(
    absl::flat_hash_map<std::string, std::string> mapping) {
  return CreateResourcesWithMapping(CreateDefaultResources(),
                                    std::move(mapping));
}

std::unique_ptr<Resources> CreateResourcesWithMapping(
    std::unique_ptr<Resources> resources,
    absl::flat_hash_map<std::string, std::string> mapping) {
  return std::make_unique<ResourcesWithMapping>(std::move(resources),
                                                std::move(mapping));
}

}  // namespace mediapipe
