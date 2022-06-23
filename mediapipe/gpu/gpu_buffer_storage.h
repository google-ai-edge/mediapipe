#ifndef MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/tool/type_util.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

namespace mediapipe {
class GpuBuffer;
namespace internal {

template <class... T>
struct types {};

template <class V>
class ViewProvider;

// Interface for a backing storage for GpuBuffer.
class GpuBufferStorage {
 public:
  virtual ~GpuBufferStorage() = default;
  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual GpuBufferFormat format() const = 0;
  // We can't use dynamic_cast since we want to support building without RTTI.
  // The public methods delegate to the type-erased private virtual method.
  template <class T>
  T* down_cast() {
    return static_cast<T*>(const_cast<void*>(down_cast(kTypeId<T>)));
  }
  template <class T>
  const T* down_cast() const {
    return static_cast<const T*>(down_cast(kTypeId<T>));
  }

  bool can_down_cast_to(TypeId to) const { return down_cast(to) != nullptr; }
  virtual TypeId storage_type() const = 0;

 private:
  virtual const void* down_cast(TypeId to) const = 0;
};

// Used to disambiguate between overloads by manually specifying their priority.
// Higher Ns will be picked first. The caller should pass overload_priority<M>
// where M is >= the largest N used in overloads (e.g. 10).
template <int N>
struct overload_priority : public overload_priority<N - 1> {};
template <>
struct overload_priority<0> {};

// Manages the available GpuBufferStorage implementations. The list of available
// implementations is built at runtime using a registration mechanism, so that
// it can be determined by the program's dependencies.
class GpuBufferStorageRegistry {
 public:
  struct RegistryToken {};

  using StorageFactory = std::function<std::shared_ptr<GpuBufferStorage>(
      int, int, GpuBufferFormat)>;
  using StorageConverter = std::function<std::shared_ptr<GpuBufferStorage>(
      std::shared_ptr<GpuBufferStorage>)>;

  static GpuBufferStorageRegistry& Get() {
    static NoDestructor<GpuBufferStorageRegistry> registry;
    return *registry;
  }

  template <class Storage>
  RegistryToken Register() {
    return Register(
        [](int width, int height,
           GpuBufferFormat format) -> std::shared_ptr<Storage> {
          return CreateStorage<Storage>(overload_priority<10>{}, width, height,
                                        format);
        },
        Storage::GetProviderTypes());
  }

  template <class StorageFrom, class StorageTo, class F>
  RegistryToken RegisterConverter(F&& converter) {
    return Register(
        [converter](std::shared_ptr<GpuBufferStorage> source)
            -> std::shared_ptr<GpuBufferStorage> {
          return converter(std::static_pointer_cast<StorageFrom>(source));
        },
        StorageTo::GetProviderTypes(), kTypeId<StorageFrom>);
  }

  // Returns a factory function for a storage that implements
  // view_provider_type.
  StorageFactory StorageFactoryForViewProvider(TypeId view_provider_type);

  // Returns a conversion function that, given a storage of
  // existing_storage_type, converts its contents to a new storage that
  // implements view_provider_type.
  StorageConverter StorageConverterForViewProvider(
      TypeId view_provider_type, TypeId existing_storage_type);

 private:
  template <class Storage, class... Args>
  static auto CreateStorage(overload_priority<1>, Args... args)
      -> decltype(Storage::Create(args...)) {
    return Storage::Create(args...);
  }

  template <class Storage, class... Args>
  static auto CreateStorage(overload_priority<0>, Args... args) {
    return std::make_shared<Storage>(args...);
  }

  RegistryToken Register(StorageFactory factory,
                         std::vector<TypeId> provider_hashes);
  RegistryToken Register(StorageConverter converter,
                         std::vector<TypeId> provider_hashes,
                         TypeId source_storage);

  absl::flat_hash_map<TypeId, StorageFactory> factory_for_view_provider_;
  absl::flat_hash_map<std::pair<TypeId, TypeId>, StorageConverter>
      converter_for_view_provider_and_existing_storage_;
};

// Defining a member of this type causes P to be ODR-used, which forces its
// instantiation if it's a static member of a template.
template <auto* P>
struct ForceStaticInstantiation {
#ifdef _MSC_VER
  // Just having it as the template argument does not count as a use for
  // MSVC.
  static constexpr bool Use() { return P != nullptr; }
  char force_static[Use()];
#endif  // _MSC_VER
};

// T: storage type
// U...: ViewProvider<SomeView>
template <class T, class... U>
class GpuBufferStorageImpl : public GpuBufferStorage, public U... {
 public:
  static const std::vector<TypeId>& GetProviderTypes() {
    static std::vector<TypeId> kHashes{kTypeId<U>...};
    return kHashes;
  }

 private:
  virtual const void* down_cast(TypeId to) const override {
    return down_cast_impl(to, types<T, U...>{});
  }
  TypeId storage_type() const override { return kTypeId<T>; }

  const void* down_cast_impl(TypeId to, types<>) const { return nullptr; }
  template <class V, class... W>
  const void* down_cast_impl(TypeId to, types<V, W...>) const {
    if (to == kTypeId<V>) return static_cast<const V*>(this);
    return down_cast_impl(to, types<W...>{});
  }

  inline static auto registration =
      GpuBufferStorageRegistry::Get().Register<T>();
  using RequireStatics = ForceStaticInstantiation<&registration>;
};

#if !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
// This function can be overridden to enable construction of a GpuBuffer from
// platform-specific types without having to expose that type in the GpuBuffer
// definition. It is only needed for backward compatibility reasons; do not add
// overrides for new types.
std::shared_ptr<internal::GpuBufferStorage> AsGpuBufferStorage();
#endif  // !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace internal
}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_STORAGE_H_
