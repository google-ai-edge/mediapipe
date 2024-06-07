#include "mediapipe/framework/graph_service_manager.h"

#include "absl/synchronization/mutex.h"

namespace mediapipe {

absl::Status GraphServiceManager::SetServicePacket(
    const GraphServiceBase& service, Packet p) {
  // TODO: check service is already set?
  absl::MutexLock lock(&service_packets_mutex_);
  service_packets_[service.key] = std::move(p);
  return absl::OkStatus();
}

Packet GraphServiceManager::GetServicePacket(
    const GraphServiceBase& service) const {
  absl::MutexLock lock(&service_packets_mutex_);
  auto it = service_packets_.find(service.key);
  if (it == service_packets_.end()) {
    return {};
  }
  return it->second;
}

}  // namespace mediapipe
