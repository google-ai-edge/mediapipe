#include <atomic>
#include "OlaGraph.hpp"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/graph_service.h"

#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#if defined(__APPLE__)
#include "mediapipe/gpu/MPPGraphGPUData.h"
#include "mediapipe/objc/util.h"
#endif



namespace Opipe {

void CallFrameDelegate(void* wrapperVoid, const std::string& streamName,
                       MPPPacketType packetType,
                       const mediapipe::Packet& packet) {
}

OlaGraph::OlaGraph(const mediapipe::CalculatorGraphConfig &config) {
    _config = config;
    _graph = absl::make_unique<mediapipe::CalculatorGraph>();
    
}

OlaGraph::~OlaGraph() {
    
}

mediapipe::ProfilingContext OlaGraph::getProfiler() {
    return _graph->getProfiler();
}

void OlaGraph::setHeaderPacket(const mediapipe::Packet &packet, std::string streamName) {
    _streamHeaders[streamName] = packet;
}

void OlaGraph::setSidePacket(const mediapipe::Packet &packet, std::string name) {
    _inputSidePackets[name] = packet;
}

void OlaGraph::setServicePacket(mediapipe::Packet &packet,const mediapipe::GraphServiceBase &service) {
    
}

void OlaGraph::addSidePackets(const std::map<std::string, mediapipe::Packet> &extraSidePackets) {
    
}

void OlaGraph::addFrameOutputStream(const std::string &outputStreamName,
                                    MPPPacketType packetType) {
    
}

bool OlaGraph::start() {
    return false;
}

bool OlaGraph::sendPacket(const mediapipe::Packet &packet,
                          const std::string &streamName) {
    return false;
}

bool OlaGraph::movePacket(mediapipe::Packet &&packet, const std::string &streamName) {
    return false;
}

/// Sets the maximum queue size for a stream. Experimental feature, currently
/// only supported for graph input streams. Should be called before starting the
/// graph.
bool OlaGraph::setMaxQueueSize(int maxQueueSize,
                               const std::string &streamName) {
    
}

#if defined(__APPLE__)
/// Creates a MediaPipe packet wrapping the given pixelBuffer;
mediapipe::Packet OlaGraph::packetWithPixelBuffer(CVPixelBufferRef pixelBuffer,
                                                  MPPPacketType packetType) {
    return 0;
}



/// Creates a MediaPipe packet of type Image, wrapping the given CVPixelBufferRef.
mediapipe::Packet OlaGraph::imagePacketWithPixelBuffer(CVPixelBufferRef pixelBuffer) {
    return 0;
}

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. Drops frames and
/// returns NO if maxFramesInFlight is exceeded. If allowOverwrite is set to YES,
/// allows MediaPipe to overwrite the packet contents on successful sending for
/// possibly increased efficiency. Returns YES if the packet was successfully sent.
bool OlaGraph::sendPixelBuffer(CVPixelBufferRef imageBuffer,
                               const std::string & inputName,
                               MPPPacketType packetType,
                               const mediapipe::Timestamp &timestamp,
                               bool allowOverwrite) {
    return false;
}

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. Drops frames and
/// returns NO if maxFramesInFlight is exceeded. Returns YES if the packet was
/// successfully sent.
bool OlaGraph::sendPixelBuffer(CVPixelBufferRef imageBuffer,
                               const std::string & inputName,
                               MPPPacketType packetType,
                               const mediapipe::Timestamp &timestamp) {
    return false;
}

/// Sends a pixel buffer into a graph input stream, using the specified packet
/// type. The graph must have been started before calling this. The timestamp is
/// automatically incremented from the last timestamp used by this method. Drops
/// frames and returns NO if maxFramesInFlight is exceeded. Returns YES if the
/// packet was successfully sent.
bool OlaGraph::sendPixelBuffer(CVPixelBufferRef imageBuffer,
                               const std::string & inputName,
                               MPPPacketType packetType) {
    return false;
}

#endif

/// Cancels a graph run. You must still call waitUntilDoneWithError: after this.
void OlaGraph::cancel() {
    
}

/// Check if the graph contains this input stream
bool OlaGraph::hasInputStream(const std::string &inputName) {
    return false;
}

/// Closes an input stream.
/// You must close all graph input streams before stopping the graph.
/// @return YES if successful.
bool OlaGraph::closeInputStream(const std::string &inputName) {
    return false;
}

/// Closes all graph input streams.
/// @return YES if successful.
bool OlaGraph::closeAllInputStreams() {
    return false;
}

/// Stops running the graph.
/// Call this before releasing this object. All input streams must have been
/// closed. This call does not time out, so you should not call it from the main
/// thread.
/// @return YES if successful.
bool OlaGraph::waitUntilDone() {
    return false;
}

/// Waits for the graph to become idle.
bool OlaGraph::waitUntilIdle() {
    return false;
}


}
