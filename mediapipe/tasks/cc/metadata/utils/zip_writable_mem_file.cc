/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/metadata/utils/zip_writable_mem_file.h"

#include <algorithm>
#include <cstdio>

#include "absl/strings/string_view.h"
#include "contrib/minizip/ioapi.h"

namespace mediapipe {
namespace tasks {
namespace metadata {

ZipWritableMemFile::ZipWritableMemFile(const char* buffer, size_t size)
    : data_(buffer, size), offset_(0) {
  zlib_filefunc64_def_.zopen64_file = OpenFile;
  zlib_filefunc64_def_.zread_file = ReadFile;
  zlib_filefunc64_def_.zwrite_file = WriteFile;
  zlib_filefunc64_def_.ztell64_file = TellFile;
  zlib_filefunc64_def_.zseek64_file = SeekFile;
  zlib_filefunc64_def_.zclose_file = CloseFile;
  zlib_filefunc64_def_.zerror_file = ErrorFile;
  zlib_filefunc64_def_.opaque = this;
}

zlib_filefunc64_def& ZipWritableMemFile::GetFileFunc64Def() {
  return zlib_filefunc64_def_;
}

absl::string_view ZipWritableMemFile::GetFileContent() const { return data_; }

/* static */
voidpf ZipWritableMemFile::OpenFile(voidpf opaque, const void* filename,
                                    int mode) {
  // Result is never used, but needs to be non-null for `zipOpen2` not to fail.
  return opaque;
}

/* static */
uLong ZipWritableMemFile::ReadFile(voidpf opaque, voidpf stream, void* buf,
                                   uLong size) {
  auto* mem_file = static_cast<ZipWritableMemFile*>(opaque);
  if (mem_file->offset_ < 0 || mem_file->Size() < mem_file->offset_) {
    return 0;
  }
  if (mem_file->offset_ + size > mem_file->Size()) {
    size = mem_file->Size() - mem_file->offset_;
  }
  memcpy(buf,
         static_cast<const char*>(mem_file->data_.c_str()) + mem_file->offset_,
         size);
  mem_file->offset_ += size;
  return size;
}

/* static */
uLong ZipWritableMemFile::WriteFile(voidpf opaque, voidpf stream,
                                    const void* buf, uLong size) {
  auto* mem_file = static_cast<ZipWritableMemFile*>(opaque);
  if (mem_file->offset_ + size > mem_file->Size()) {
    mem_file->data_.resize(mem_file->offset_ + size);
  }
  mem_file->data_.replace(mem_file->offset_, size,
                          static_cast<const char*>(buf), size);
  mem_file->offset_ += size;
  return size;
}

/* static */
ZPOS64_T ZipWritableMemFile::TellFile(voidpf opaque, voidpf stream) {
  return static_cast<ZipWritableMemFile*>(opaque)->offset_;
}

/* static */
long ZipWritableMemFile::SeekFile  // NOLINT
    (voidpf opaque, voidpf stream, ZPOS64_T offset, int origin) {
  auto* mem_file = static_cast<ZipWritableMemFile*>(opaque);
  switch (origin) {
    case SEEK_SET:
      mem_file->offset_ = offset;
      return 0;
    case SEEK_CUR:
      if (mem_file->offset_ + offset < 0 ||
          mem_file->offset_ + offset > mem_file->Size()) {
        return -1;
      }
      mem_file->offset_ += offset;
      return 0;
    case SEEK_END:
      if (mem_file->Size() - offset < 0 ||
          mem_file->Size() - offset > mem_file->Size()) {
        return -1;
      }
      mem_file->offset_ = offset + mem_file->Size();
      return 0;
    default:
      return -1;
  }
}

/* static */
int ZipWritableMemFile::CloseFile(voidpf opaque, voidpf stream) {
  // Nothing to do.
  return 0;
}

/* static */
int ZipWritableMemFile::ErrorFile(voidpf opaque, voidpf stream) {
  // Unused.
  return 0;
}

}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
