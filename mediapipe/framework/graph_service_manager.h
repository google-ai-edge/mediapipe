#ifndef MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_
#define MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_

#include <memory>

#include "absl/status/status.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/packet.h"

namespace mediapipe {

class GraphServiceManager {
 public:
  template <typename T>
  absl::Status SetServiceObject(const GraphService<T>& service,
                                std::shared_ptr<T> object) {
    return SetServicePacket(service,
                            MakePacket<std::shared_ptr<T>>(std::move(object)));
  }

  absl::Status SetServicePacket(const GraphServiceBase& service, Packet p);

  template <typename T>
  std::shared_ptr<T> GetServiceObject(const GraphService<T>& service) const {
    Packet p = GetServicePacket(service);
    if (p.IsEmpty()) return nullptr;
    return p.Get<std::shared_ptr<T>>();
  }

  const std::map<std::string, Packet>& ServicePackets() {
    return service_packets_;
  }

 private:
  Packet GetServicePacket(const GraphServiceBase& service) const;

  std::map<std::string, Packet> service_packets_;

  friend class CalculatorGraph;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_
