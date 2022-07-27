#include <atomic>
#include "ola_graph.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/graph_service.h"

#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"


using namespace mediapipe;

namespace Opipe
{

    void CallFrameDelegate(void *wrapperVoid, const std::string &streamName, MPPPacketType packetType,
                           const mediapipe::Packet &packet)
    {
        OlaGraph *graph = (OlaGraph *)wrapperVoid;
        if (graph->_delegate.expired())
        {
            return;
        }
        
        graph->_delegate.lock()->outputPacket(graph, packet, streamName);
        
        if (packetType == MPPPacketTypeRaw)
        {
            graph->_delegate.lock()->outputPacket(graph, packet, packetType, streamName);
        }
#if defined(__APPLE__)
        else if (packetType == MPPPacketTypePixelBuffer ||
                 packetType == MPPPacketTypeImage)
        {
            graph->_framesInFlight--;
            CVPixelBufferRef pixelBuffer;
            if (packetType == MPPPacketTypePixelBuffer)
                pixelBuffer = mediapipe::GetCVPixelBufferRef(packet.Get<mediapipe::GpuBuffer>());
            else
                pixelBuffer = packet.Get<mediapipe::Image>().GetCVPixelBufferRef();

            graph->_delegate.lock()->outputPixelbuffer(graph, pixelBuffer, streamName, packet.Timestamp().Value());
#endif
        }
        
        
    }

    OlaGraph::OlaGraph(const mediapipe::CalculatorGraphConfig &config)
    {
        _config = config;
        _graph = absl::make_unique<mediapipe::CalculatorGraph>();
    }

    OlaGraph::~OlaGraph()
    {
    }

    mediapipe::ProfilingContext *OlaGraph::getProfiler()
    {
        return _graph->profiler();
    }

    void OlaGraph::setHeaderPacket(const mediapipe::Packet &packet, std::string streamName)
    {
        _streamHeaders[streamName] = packet;
    }

    void OlaGraph::setSidePacket(const mediapipe::Packet &packet, std::string name)
    {
        _inputSidePackets[name] = packet;
    }

    void OlaGraph::setServicePacket(mediapipe::Packet &packet, const mediapipe::GraphServiceBase &service)
    {
        _servicePackets[&service] = std::move(packet);
    }

    void OlaGraph::addSidePackets(const std::map<std::string, mediapipe::Packet> &extraSidePackets)
    {
        _inputSidePackets.insert(extraSidePackets.begin(), extraSidePackets.end());
    }

    void OlaGraph::addFrameOutputStream(const std::string &outputStreamName,
                                        MPPPacketType packetType)
    {
        std::string callbackInputName;
        mediapipe::tool::AddCallbackCalculator(outputStreamName, &_config, &callbackInputName,
                                               /*use_std_function=*/true);
        // No matter what ownership qualifiers are put on the pointer, NewPermanentCallback will
        // still end up with a strong pointer to MPPGraph*. That is why we use void* instead.
        void *wrapperVoid = this;
        _inputSidePackets[callbackInputName] =
            mediapipe::MakePacket<std::function<void(const mediapipe::Packet &)>>([wrapperVoid, outputStreamName, packetType](const mediapipe::Packet &packet)
                                                                                  { CallFrameDelegate(wrapperVoid, outputStreamName, packetType, packet); });
    }

    bool OlaGraph::start()
    {
        absl::Status status = performStart();
        _started = status.ok();
        return status.ok();
    }

    absl::Status OlaGraph::performStart()
    {
        absl::Status status = _graph->Initialize(_config);
        if (!status.ok())
        {
            return status;
        }
        for (const auto &service_packet : _servicePackets)
        {
            status = _graph->SetServicePacket(*service_packet.first, service_packet.second);
            if (!status.ok())
            {
                return status;
            }
        }
        status = _graph->StartRun(_inputSidePackets, _streamHeaders);
        NSLog(@"errors:%@", [NSString stringWithUTF8String:status.ToString().c_str()]);
        if (!status.ok())
        {
            return status;
        }
        return status;
    }

    bool OlaGraph::sendPacket(const mediapipe::Packet &packet,
                              const std::string &streamName)
    {
        absl::Status status = _graph->AddPacketToInputStream(streamName, packet);
        NSLog(@"errors:%@", [NSString stringWithUTF8String:status.ToString().c_str()]);
        return status.ok();
    }

    bool OlaGraph::movePacket(mediapipe::Packet &&packet, const std::string &streamName)
    {
        absl::Status status = _graph->AddPacketToInputStream(streamName, std::move(packet));
        NSLog(@"errors:%@", [NSString stringWithUTF8String:status.ToString().c_str()]);
        return status.ok();
    }

    /// Sets the maximum queue size for a stream. Experimental feature, currently
    /// only supported for graph input streams. Should be called before starting the
    /// graph.
    bool OlaGraph::setMaxQueueSize(int maxQueueSize,
                                   const std::string &streamName)
    {
        absl::Status status = _graph->SetInputStreamMaxQueueSize(streamName, maxQueueSize);
        return status.ok();
    }

#if defined(__APPLE__)
    /// Creates a MediaPipe packet wrapping the given pixelBuffer;
    mediapipe::Packet OlaGraph::packetWithPixelBuffer(CVPixelBufferRef pixelBuffer,
                                                      MPPPacketType packetType)
    {
        mediapipe::Packet packet;
        if (packetType == MPPPacketTypeImageFrame || packetType == MPPPacketTypeImageFrameBGRANoSwap)
        {
            auto frame = CreateImageFrameForCVPixelBuffer(
                pixelBuffer, /* canOverwrite = */ false,
                /* bgrAsRgb = */ packetType == MPPPacketTypeImageFrameBGRANoSwap);
            packet = mediapipe::Adopt(frame.release());
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
        }
        else if (packetType == MPPPacketTypePixelBuffer)
        {
            packet = mediapipe::MakePacket<mediapipe::GpuBuffer>(pixelBuffer);
#endif // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
        }
        else if (packetType == MPPPacketTypeImage)
        {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
            // GPU
            packet = mediapipe::MakePacket<mediapipe::Image>(pixelBuffer);
#else
            // CPU
            auto frame = CreateImageFrameForCVPixelBuffer(imageBuffer, /* canOverwrite = */ false,
                                                          /* bgrAsRgb = */ false);
            packet = mediapipe::MakePacket<mediapipe::Image>(std::move(frame));
#endif // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
        }
        else
        {
            assert(false);
        }
        return packet;
    }

    /// Creates a MediaPipe packet of type Image, wrapping the given CVPixelBufferRef.
    mediapipe::Packet OlaGraph::imagePacketWithPixelBuffer(CVPixelBufferRef pixelBuffer)
    {
        return packetWithPixelBuffer(pixelBuffer, MPPPacketTypeImage);
    }

    /// Sends a pixel buffer into a graph input stream, using the specified packet
    /// type. The graph must have been started before calling this. Drops frames and
    /// returns NO if maxFramesInFlight is exceeded. If allowOverwrite is set to YES,
    /// allows MediaPipe to overwrite the packet contents on successful sending for
    /// possibly increased efficiency. Returns YES if the packet was successfully sent.
    bool OlaGraph::sendPixelBuffer(CVPixelBufferRef imageBuffer,
                                   const std::string &inputName,
                                   MPPPacketType packetType,
                                   const mediapipe::Timestamp &timestamp,
                                   bool allowOverwrite)
    {
        if (_maxFramesInFlight && _framesInFlight >= _maxFramesInFlight)
            return false;
        mediapipe::Packet packet = packetWithPixelBuffer(imageBuffer, packetType);
        bool success;
        if (allowOverwrite)
        {
            packet = std::move(packet).At(timestamp);
            success = movePacket(std::move(packet), inputName);
        }
        else
        {
            success = sendPacket(packet.At(timestamp), inputName);
        }
        if (success)
            _framesInFlight++;
        return success;
    }

    /// Sends a pixel buffer into a graph input stream, using the specified packet
    /// type. The graph must have been started before calling this. Drops frames and
    /// returns NO if maxFramesInFlight is exceeded. Returns YES if the packet was
    /// successfully sent.
    bool OlaGraph::sendPixelBuffer(CVPixelBufferRef imageBuffer,
                                   const std::string &inputName,
                                   MPPPacketType packetType,
                                   const mediapipe::Timestamp &timestamp)
    {
        return sendPixelBuffer(imageBuffer, inputName, packetType, timestamp, false);
    }

    /// Sends a pixel buffer into a graph input stream, using the specified packet
    /// type. The graph must have been started before calling this. The timestamp is
    /// automatically incremented from the last timestamp used by this method. Drops
    /// frames and returns NO if maxFramesInFlight is exceeded. Returns YES if the
    /// packet was successfully sent.
    bool OlaGraph::sendPixelBuffer(CVPixelBufferRef imageBuffer,
                                   const std::string &inputName,
                                   MPPPacketType packetType)
    {
        if (_frameTimestamp < mediapipe::Timestamp::Min())
        {
            _frameTimestamp = mediapipe::Timestamp::Min();
        }
        else
        {
            _frameTimestamp++;
        }
        return sendPixelBuffer(imageBuffer, inputName, packetType, _frameTimestamp);
    }

#endif

    /// Cancels a graph run. You must still call waitUntilDoneWithError: after this.
    void OlaGraph::cancel()
    {
        _graph->Cancel();
    }

    /// Check if the graph contains this input stream
    bool OlaGraph::hasInputStream(const std::string &inputName)
    {
        return _graph->HasInputStream(inputName);
    }

    /// Closes an input stream.
    /// You must close all graph input streams before stopping the graph.
    /// @return YES if successful.
    bool OlaGraph::closeInputStream(const std::string &inputName)
    {
        absl::Status status = _graph->CloseInputStream(inputName);
        return status.ok();
    }

    /// Closes all graph input streams.
    /// @return YES if successful.
    bool OlaGraph::closeAllInputStreams()
    {
        absl::Status status = _graph->CloseAllInputStreams();
        return status.ok();
    }

    /// Stops running the graph.
    /// Call this before releasing this object. All input streams must have been
    /// closed. This call does not time out, so you should not call it from the main
    /// thread.
    /// @return YES if successful.
    bool OlaGraph::waitUntilDone()
    {
        absl::Status status = _graph->WaitUntilDone();
        _started = false;
        return status.ok();
    }

    /// Waits for the graph to become idle.
    bool OlaGraph::waitUntilIdle()
    {
        absl::Status status = _graph->WaitUntilIdle();
        return status.ok();
    }

}
