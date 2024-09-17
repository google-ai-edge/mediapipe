#include "mediapipe/framework/resources.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

ABSL_DECLARE_FLAG(std::string, resource_root_dir);

namespace mediapipe {
namespace {

TEST(Resources, CanCreateStringResource) {
  std::unique_ptr<Resource> resource = MakeStringResource("Test string.");
  EXPECT_EQ(resource->ToStringView(), "Test string.");
}

TEST(Resources, CanCreateNoCleanupResource) {
  std::string data = "Test string.";
  std::unique_ptr<Resource> resource =
      MakeNoCleanupResource(data.data(), data.size());
  EXPECT_EQ(resource->ToStringView(), "Test string.");
}

TEST(Resources, CanCreateDefaultResourcesAndReadFileContents) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Resource> resource,
      resources->Get("mediapipe/framework/testdata/resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
}

TEST(Resources, CanReadFileContentsByUnresolvedId) {
  absl::SetFlag(&FLAGS_resource_root_dir, "mediapipe/framework/testdata");
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Resource> resource,
                          resources->Get("resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
}

// PathToResourceFile is called in many in places and Resource object may
// receive an already resolved id.
TEST(Resources, CanReadFileContentsByResolvedIdWhenRootDirSpecified) {
  absl::SetFlag(&FLAGS_resource_root_dir, "mediapipe/framework/testdata");
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Resource> resource,
      resources->Get("mediapipe/framework/testdata/resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
}

TEST(Resources, CanCreateDefaultResourcesWithMappingAndReadFileContents) {
  absl::flat_hash_map<std::string, std::string> mapping = {
      {"$CUSTOM_ID", "mediapipe/framework/testdata/resource_calculator.data"}};
  std::unique_ptr<Resources> resources =
      CreateDefaultResourcesWithMapping(std::move(mapping));

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Resource> resource,
                          resources->Get("$CUSTOM_ID"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
}

class CustomResources : public Resources {
 public:
  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const final {
    if (resource_id == "custom/resource/id") {
      return MakeStringResource("Custom content.");
    }
    return default_resources_->Get(resource_id);
  }

 private:
  std::unique_ptr<Resources> default_resources_ = CreateDefaultResources();
};

TEST(Resources, CanCreateCustomResourcesAndReuseDefault) {
  std::unique_ptr<Resources> resources = std::make_unique<CustomResources>();

  std::unique_ptr<Resource> resource;
  MP_ASSERT_OK_AND_ASSIGN(
      resource,
      resources->Get("mediapipe/framework/testdata/resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
  MP_ASSERT_OK_AND_ASSIGN(resource, resources->Get("custom/resource/id"));
  EXPECT_EQ(resource->ToStringView(), "Custom content.");
}

TEST(Resources, CanCreateCustomResourcesAndUseMapping) {
  std::unique_ptr<Resources> resources = std::make_unique<CustomResources>();
  absl::flat_hash_map<std::string, std::string> mapping = {
      {"$CUSTOM_ID", "custom/resource/id"}};
  resources =
      CreateResourcesWithMapping(std::move(resources), std::move(mapping));

  std::unique_ptr<Resource> resource;
  MP_ASSERT_OK_AND_ASSIGN(
      resource,
      resources->Get("mediapipe/framework/testdata/resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
  MP_ASSERT_OK_AND_ASSIGN(resource, resources->Get("$CUSTOM_ID"));
  EXPECT_EQ(resource->ToStringView(), "Custom content.");
}

TEST(Resource, StringResourceCanBeConsumed) {
  constexpr absl::string_view kData = "contents";
  std::string data(kData);
  std::unique_ptr<Resource> resource = MakeStringResource(std::move(data));
  std::string consumed_str = std::move(*resource).ReleaseOrCopyAsString();
  EXPECT_EQ(consumed_str, kData);
}

}  // namespace
}  // namespace mediapipe
