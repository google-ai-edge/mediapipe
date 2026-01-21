// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/deps/file_helpers.h"

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <handleapi.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#endif  // _WIN32
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <cerrno>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/config.h"
#include "absl/cleanup/cleanup.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/deps/mmapped_file.h"
#include "mediapipe/framework/deps/platform_strings.h"  // IWYU pragma: keep
#ifndef _WIN32
#include "mediapipe/framework/formats/unique_fd.h"
#endif  // !_WIN32
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {
namespace file {
namespace {
size_t RoundUp(size_t size, size_t align) {
  return (size + align - 1) & ~(align - 1);
}

// Helper class that returns all entries (files, directories) in a directory,
// except "." and "..". Example usage:
//
// DirectoryListing listing("/tmp");
// while (listing.HasNextEntry()) {
//   std::cout << listing.NextEntry() << std::endl;
// }
#ifndef _WIN32
class DirectoryListing {
 public:
  explicit DirectoryListing(const std::string& directory) {
    dir_ = opendir(directory.c_str());
    if (dir_) {
      ReadNextEntry();
    }
  }
  ~DirectoryListing() {
    if (dir_) {
      closedir(dir_);
    }
  }

  // Returns true if another entry in this directory listing is available.
  bool HasNextEntry() { return dir_ != nullptr && next_entry_ != nullptr; }

  // Returns the next entry in this directory listing and advances to the entry
  // after the one that is returned, if it exists.
  std::string NextEntry() {
    if (HasNextEntry()) {
      std::string result = std::string(next_entry_->d_name);
      ReadNextEntry();
      return result;
    } else {
      return std::string();
    }
  }

 private:
  void ReadNextEntry() {
    next_entry_ = readdir(dir_);
    while (next_entry_ && (std::string(next_entry_->d_name) == "." ||
                           std::string(next_entry_->d_name) == "..")) {
      next_entry_ = readdir(dir_);
    }
  }

  DIR* dir_ = nullptr;
  struct dirent* next_entry_ = nullptr;
};
#else
class DirectoryListing {
 public:
  explicit DirectoryListing(const std::string& directory)
      : directory_(Utf8ToNative(directory)) {
    PlatformString search_string =
        directory_ + PLATFORM_STRING_LITERAL("\\*.*");
    find_handle_ = FindFirstFile(search_string.c_str(), &find_data_);
  }

  ~DirectoryListing() {
    if (find_handle_ != INVALID_HANDLE_VALUE) {
      FindClose(find_handle_);
    }
  }

  // Returns true if another entry in this directory listing is available.
  bool HasNextEntry() { return find_handle_ != INVALID_HANDLE_VALUE; }

  // Returns the next entry in this directory listing and advances to the entry
  // after the one that is returned, if it exists.
  std::string NextEntry() {
    if (HasNextEntry()) {
      PlatformString result = directory_ + PLATFORM_STRING_LITERAL("\\") +
                              PlatformString(find_data_.cFileName);
      ReadNextEntry();
      return NativeToUtf8(result);
    } else {
      return std::string();
    }
  }

 private:
  void ReadNextEntry() {
    int find_result = FindNextFile(find_handle_, &find_data_);
    while (find_result != 0 && (PlatformString(find_data_.cFileName) ==
                                    PLATFORM_STRING_LITERAL(".") ||
                                PlatformString(find_data_.cFileName) ==
                                    PLATFORM_STRING_LITERAL(".."))) {
      find_result = FindNextFile(find_handle_, &find_data_);
    }

    if (find_result == 0) {
      FindClose(find_handle_);
      find_handle_ = INVALID_HANDLE_VALUE;
    }
  }

  const PlatformString directory_;
  HANDLE find_handle_ = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATA find_data_;
};
#endif  // _WIN32

}  // namespace

absl::Status GetContents(absl::string_view path, std::string* output,
                         bool read_as_binary) {
  FILE* fp = fopen(std::string(path).c_str(), read_as_binary ? "rb" : "r");
  if (fp == NULL) {
    return absl::NotFoundError(absl::StrCat("Can't find file: ", path));
  }

  output->clear();
  while (!feof(fp)) {
    char buf[4096];
    size_t ret = fread(buf, 1, 4096, fp);
    if (ret == 0 && ferror(fp)) {
      return absl::UnavailableError(
          absl::StrCat("Error while reading file: ", path));
    }
    output->append(std::string(buf, ret));
  }
  fclose(fp);
  return absl::OkStatus();
}

absl::Status SetContents(absl::string_view path, absl::string_view content) {
  FILE* fp = fopen(std::string(path).c_str(), "wb");
  if (fp == NULL) {
    return absl::InvalidArgumentError(absl::StrCat("Can't open file: ", path));
  }

  fwrite(content.data(), sizeof(char), content.size(), fp);
  size_t write_error = ferror(fp);
  if (fclose(fp) != 0 || write_error) {
    return absl::UnavailableError(
        absl::StrCat("Error while writing file: ", path,
                     ". Error message: ", strerror(write_error)));
  }
  return absl::OkStatus();
}

absl::Status AppendStringToFile(absl::string_view path,
                                absl::string_view contents) {
  FILE* fp = fopen(std::string(path).c_str(), "ab");
  if (!fp) {
    return absl::InvalidArgumentError(absl::StrCat("Can't open file: ", path));
  }

  fwrite(contents.data(), sizeof(char), contents.size(), fp);
  size_t write_error = ferror(fp);
  if (fclose(fp) != 0 || write_error) {
    return absl::UnavailableError(
        absl::StrCat("Error while writing file: ", path,
                     ". Error message: ", strerror(write_error)));
  }
  return absl::OkStatus();
}

#ifdef _WIN32
class WindowsMMap : public MemoryMappedFile {
 public:
  WindowsMMap(std::string path, const void* base_address, size_t length,
              HANDLE file_handle, HANDLE mapping_handle)
      : MemoryMappedFile(std::move(path), base_address, length),
        file_handle_(file_handle),
        mapping_handle_(mapping_handle) {}

  virtual absl::Status Close() override;

 private:
  const HANDLE file_handle_;
  const HANDLE mapping_handle_;
};

absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MMapFile(
    absl::string_view path) {
  std::string name_string = std::string(path);
  const HANDLE file_handle = CreateFile(
      /*lpFileName=*/Utf8ToNative(name_string).c_str(),
      /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_READ,
      /*lpSecurityAttributes=*/NULL,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_ATTRIBUTE_NORMAL,
      /*hTemplateFile=*/NULL);
  if (file_handle == INVALID_HANDLE_VALUE) {
    return absl::UnavailableError(
        absl::StrCat("Failed to open the file '", path,
                     "' for reading: ", FormatLastError()));
  }
  absl::Cleanup file_closer = [file_handle] { CloseHandle(file_handle); };

  // We're calling `CreateFileMappingA` regardless of `UNICODE` because we don't
  // pass the `lpName` string parameter.
  const HANDLE mapping_handle = CreateFileMappingA(
      /*hFile=*/file_handle,
      /*lpFileMappingAttributes=*/NULL,
      /*flProtect=*/PAGE_READONLY,
      /*dwMaximumSizeHigh=*/0,  // If `dwMaximumSize{Low,High} are zero,
      /*dwMaximumSizeLow=*/0,   // the maximum mapping size is the file size.
      /*lpName=*/NULL);
  if (mapping_handle == INVALID_HANDLE_VALUE) {
    return absl::UnavailableError(
        absl::StrCat("Failed to create a memory mapping for the file '", path,
                     "': ", FormatLastError()));
  }
  absl::Cleanup mapping_closer = [mapping_handle] {
    CloseHandle(mapping_handle);
  };

  const LPVOID base_address = MapViewOfFile(
      /*hFileMappingObject=*/mapping_handle,
      /*dwDesiredAccess=*/FILE_MAP_READ,
      /*dwFileOffsetHigh=*/0,
      /*dwFileOffsetLow=*/0,
      /*dwNumberOfBytesToMap=*/0  // Extends to the file end.
  );
  if (base_address == NULL) {
    return absl::UnavailableError(absl::StrCat(
        "Failed to memory-map the file '", path, "': ", FormatLastError()));
  }

  LARGE_INTEGER large_length;
  const BOOL success = GetFileSizeEx(file_handle, &large_length);
  if (!success) {
    return absl::UnavailableError(
        absl::StrCat("Failed to determine the size of the file '", path,
                     "': ", FormatLastError()));
  }
  const size_t length = static_cast<size_t>(large_length.QuadPart);

  std::move(file_closer).Cancel();
  std::move(mapping_closer).Cancel();

  return std::make_unique<WindowsMMap>(std::move(name_string), base_address,
                                       length, file_handle, mapping_handle);
}

absl::Status WindowsMMap::Close() {
  BOOL success = UnmapViewOfFile(BaseAddress());
  if (!success) {
    return absl::UnavailableError(absl::StrCat(
        "Failed to unmap the file '", Path(), "': ", FormatLastError()));
  }
  success = CloseHandle(mapping_handle_);
  if (!success) {
    return absl::UnavailableError(
        absl::StrCat("Failed to close the memory mapping for file '", Path(),
                     "': ", FormatLastError()));
  }
  success = CloseHandle(file_handle_);
  if (!success) {
    return absl::UnavailableError(absl::StrCat(
        "Failed to close the file '", Path(), "': ", FormatLastError()));
  }
  return absl::OkStatus();
}
#elif ABSL_HAVE_MMAP
class PosixMMap : public MemoryMappedFile {
 public:
  PosixMMap(std::string path, const void* base_address, size_t length,
            UniqueFd&& fd)
      : MemoryMappedFile(path, base_address, length),
        unique_fd_(std::move(fd)) {}

  absl::StatusOr<int> TryGetFd() const override { return unique_fd_.Get(); }

  absl::Status Close() override;

 private:
  UniqueFd unique_fd_;
};

absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MMapFile(
    absl::string_view path) {
  std::string name_string = std::string(path);
  const int fd = open(name_string.c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::UnavailableError(absl::StrCat(
        "Couldn't open file '", path, "' for reading: ", FormatLastError()));
  }
  UniqueFd unique_fd(fd);

  struct stat file_stat;
  const int status = fstat(unique_fd.Get(), &file_stat);
  if (status < 0) {
    return absl::UnavailableError(
        absl::StrCat("Couldn't stat file '", path, "': ", FormatLastError()));
  }
  size_t length = file_stat.st_size;

  const void* base_address =
      mmap(nullptr, length, PROT_READ, /*flags=*/MAP_SHARED, unique_fd.Get(),
           /*offset=*/0);
  if (base_address == MAP_FAILED) {
    return absl::UnavailableError(absl::StrCat(
        "Couldn't map file '", path, "' into memory: ", FormatLastError()));
  }

  return std::make_unique<PosixMMap>(std::move(name_string), base_address,
                                     length, std::move(unique_fd));
}

absl::Status PosixMMap::Close() {
  // `munmap` length should be a multiple of page size.
  const int page_size = sysconf(_SC_PAGESIZE);
  const size_t aligned_length =
      RoundUp(Length(), static_cast<size_t>(page_size));
  int status = munmap(const_cast<void*>(BaseAddress()), aligned_length);
  if (status < 0) {
    return absl::UnavailableError(absl::StrCat(
        "Couldn't unmap file '", Path(), "' from memory: ", FormatLastError()));
  }

  status = close(unique_fd_.Release());
  if (status < 0) {
    return absl::UnavailableError(absl::StrCat("Couldn't close file '", Path(),
                                               "': ", FormatLastError()));
  }
  return absl::OkStatus();
}
#else   // _WIN32 / ABSL_HAVE_MMAP
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MMapFile(
    absl::string_view path) {
  return absl::UnavailableError(absl::StrCat(
      "No supported memory-mapping mechanism is provided for file '", path,
      "'"));
}

absl::Status LockMemory(const void* base_address, size_t length) {
  return absl::UnavailableError("Locking memory unsupported");
}

absl::Status UnlockMemory(const void* base_address, size_t length) {
  return absl::UnavailableError(
      "Shouldn't attempt unlocking memory where locking is not supported");
}
#endif  // _WIN32 / ABSL_HAVE_MMAP

absl::Status MatchInTopSubdirectories(const std::string& parent_directory,
                                      const std::string& file_name,
                                      std::vector<std::string>* results) {
  DirectoryListing parent_listing(parent_directory);

  while (parent_listing.HasNextEntry()) {
    std::string subdirectory =
        JoinPath(parent_directory, parent_listing.NextEntry());
    DirectoryListing subdirectory_listing(subdirectory);
    while (subdirectory_listing.HasNextEntry()) {
      std::string next_entry = subdirectory_listing.NextEntry();
      if (absl::EndsWith(next_entry, file_name)) {
        results->push_back(JoinPath(subdirectory, next_entry));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status MatchFileTypeInDirectory(const std::string& directory,
                                      const std::string& file_suffix,
                                      std::vector<std::string>* results) {
  DirectoryListing directory_listing(directory);

  while (directory_listing.HasNextEntry()) {
    std::string next_entry = directory_listing.NextEntry();
    if (absl::EndsWith(next_entry, file_suffix)) {
      results->push_back(JoinPath(directory, next_entry));
    }
  }

  return absl::OkStatus();
}

absl::Status Exists(absl::string_view file_name) {
#ifdef _WIN32
  // Windows needs to use stat64 for >2GB files.
  struct _stat64 buffer;
  int status = _stat64(std::string(file_name).c_str(), &buffer);
#else
  struct stat buffer;
  int status = stat(std::string(file_name).c_str(), &buffer);
#endif
  if (status == 0) {
    return absl::OkStatus();
  }
  switch (errno) {
    case EACCES:
      return absl::PermissionDeniedError("Insufficient permissions.");
    default:
      return absl::NotFoundError(
          absl::StrCat("The path does not exist: ", file_name));
  }
}

absl::Status IsDirectory(absl::string_view file_name) {
  struct stat buffer;
  int status;
  status = stat(std::string(file_name).c_str(), &buffer);
  if (status == 0) {
    if ((buffer.st_mode & S_IFMT) == S_IFDIR) {
      return absl::OkStatus();
    }
    return absl::FailedPreconditionError("The path is not a directory.");
  }
  switch (errno) {
    case EACCES:
      return absl::PermissionDeniedError("Insufficient permissions.");
    default:
      return absl::NotFoundError("The path does not exist.");
  }
}

#ifndef _WIN32
int mkdir(std::string path) {
  return ::mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}
#else
int mkdir(std::string path) { return _mkdir(path.c_str()); }
#endif

absl::Status RecursivelyCreateDir(absl::string_view path) {
  if (path.empty() || Exists(path).ok()) {
    return absl::OkStatus();
  }
  auto split_path = file::SplitPath(path);
  MP_RETURN_IF_ERROR(RecursivelyCreateDir(split_path.first));
  if (mkdir(std::string(path)) != 0) {
    switch (errno) {
      case EACCES:
        return absl::PermissionDeniedError("Insufficient permissions.");
      default:
        return absl::UnavailableError("Failed to create directory.");
    }
  }
  return absl::OkStatus();
}

}  // namespace file
}  // namespace mediapipe
