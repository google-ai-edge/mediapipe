#include "mediapipe/gpu/gpu_buffer_storage.h"

namespace mediapipe {
namespace internal {

using StorageFactory = GpuBufferStorageRegistry::StorageFactory;
using StorageConverter = GpuBufferStorageRegistry::StorageConverter;
using RegistryToken = GpuBufferStorageRegistry::RegistryToken;

StorageFactory GpuBufferStorageRegistry::StorageFactoryForViewProvider(
    TypeId view_provider_type) {
  auto it = factory_for_view_provider_.find(view_provider_type);
  if (it == factory_for_view_provider_.end()) return nullptr;
  return it->second;
}

StorageConverter GpuBufferStorageRegistry::StorageConverterForViewProvider(
    TypeId view_provider_type, TypeId existing_storage_type) {
  auto it = converter_for_view_provider_and_existing_storage_.find(
      {view_provider_type, existing_storage_type});
  if (it == converter_for_view_provider_and_existing_storage_.end())
    return nullptr;
  return it->second;
}

RegistryToken GpuBufferStorageRegistry::Register(
    StorageFactory factory, std::vector<TypeId> provider_hashes) {
  // TODO: choose between multiple factories for same provider type.
  for (const auto p : provider_hashes) {
    factory_for_view_provider_[p] = factory;
  }
  return {};
}

RegistryToken GpuBufferStorageRegistry::Register(
    StorageConverter converter, std::vector<TypeId> provider_hashes,
    TypeId source_storage) {
  // TODO: choose between multiple converters for same provider type.
  for (const auto p : provider_hashes) {
    converter_for_view_provider_and_existing_storage_[{p, source_storage}] =
        converter;
  }
  return {};
}

}  // namespace internal
}  // namespace mediapipe
