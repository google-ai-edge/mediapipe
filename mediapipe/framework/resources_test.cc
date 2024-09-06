#include "mediapipe/framework/resources.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

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

TEST(Resources, CanCreateDefaultResourcesThatAndReadFileContents) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  std::string contents;
  MP_ASSERT_OK(resources->ReadContents(
      "mediapipe/framework/testdata/resource_calculator.data", contents));
  EXPECT_EQ(contents, "File system calculator contents\n");

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Resource> resource,
      resources->Get("mediapipe/framework/testdata/resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
}

class CustomResources : public Resources {
 public:
  absl::Status ReadContents(absl::string_view resource_id, std::string& output,
                            const Options& options) const final {
    if (resource_id == "custom/resource/id") {
      output = "Custom content.";
      return absl::OkStatus();
    }
    return default_resources_->ReadContents(resource_id, output, options);
  }

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

  std::string contents;
  MP_ASSERT_OK(resources->ReadContents(
      "mediapipe/framework/testdata/resource_calculator.data", contents));
  EXPECT_EQ(contents, "File system calculator contents\n");
  MP_ASSERT_OK(resources->ReadContents("custom/resource/id", contents));
  EXPECT_EQ(contents, "Custom content.");

  std::unique_ptr<Resource> resource;
  MP_ASSERT_OK_AND_ASSIGN(
      resource,
      resources->Get("mediapipe/framework/testdata/resource_calculator.data"));
  EXPECT_EQ(resource->ToStringView(), "File system calculator contents\n");
  MP_ASSERT_OK_AND_ASSIGN(resource, resources->Get("custom/resource/id"));
  EXPECT_EQ(resource->ToStringView(), "Custom content.");
}

}  // namespace
}  // namespace mediapipe
