#ifndef MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_UTILS_H_
#define MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_UTILS_H_

#include <fcntl.h>
#include <sys/mman.h>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace xnn_utils {

std::vector<float> FillXnnRoPEWeights(size_t max_seq_len, size_t num_channels);

// expect_size_bytes == 0 means don't check size.
template <typename element_type = char>
static absl::StatusOr<std::shared_ptr<element_type>> LoadBufferFromFile(
    absl::string_view file_path, bool use_mmap = true,
    size_t expect_size_bytes = 0) {
  if (use_mmap) {
    int fd = open(file_path.data(), O_RDONLY);
    RET_CHECK_GE(fd, 0) << "open " << file_path << " failed";
    auto cleanup = absl::MakeCleanup([fd] { close(fd); });

    const size_t size = lseek(fd, 0, SEEK_END);
    if (expect_size_bytes) {
      RET_CHECK_EQ(expect_size_bytes, size)
          << "File size " << size << ", expected " << expect_size_bytes
          << ", file path " << file_path;
    }

    void* data = mmap(/*addr=*/nullptr, size, /*prot=*/PROT_READ,
                      /*flags=*/MAP_SHARED, fd, /*offset=*/0);
    RET_CHECK_NE(data, MAP_FAILED);
    RET_CHECK_NE(data, nullptr);

    return std::shared_ptr<element_type>(static_cast<element_type*>(data),
                                         [](auto* p) {});
  } else {
    auto read_buffer = std::make_shared<std::string>();
    MP_RETURN_IF_ERROR(
        file::GetContents(file_path, read_buffer.get(), file::Defaults()));

    if (expect_size_bytes) {
      RET_CHECK_EQ(expect_size_bytes, read_buffer->size())
          << "File size " << read_buffer->size() << ", expected "
          << expect_size_bytes << ", file path " << file_path;
    }

    return std::shared_ptr<element_type>(
        read_buffer, reinterpret_cast<element_type*>(read_buffer->data()));
  }
}

}  // namespace xnn_utils
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_UTILS_H_
