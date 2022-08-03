#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

#ifdef __OBJC__
#import <AVFoundation/AVFoundation.h>
#endif // __OBJC__

using namespace mediapipe;
namespace Opipe
{

    enum MPPPacketType
    {
        MPPPacketTypeRaw,
        MPPPacketTypeImage,
        MPPPacketTypeImageFrame,
        MPPPacketTypeImageFrameBGRANoSwap,
#if defined(__APPLE__)
        MPPPacketTypePixelBuffer,
#endif
    };
    class OlaGraph;
    class MPPGraphDelegate
    {
    public:

        MPPGraphDelegate() {};

        ~MPPGraphDelegate() {};
#if defined(__APPLE__)
        virtual void outputPixelbuffer(OlaGraph *graph, CVPixelBufferRef pixelbuffer,
                                       const std::string &streamName,
                                       int64_t timestamp) = 0;

#endif

        virtual void outputPacket(OlaGraph *graph,
                                  const mediapipe::Packet &packet,
                                  MPPPacketType packetType,
                                  const std::string &streamName) = 0;
        
        virtual void outputPacket(OlaGraph *graph,
                                  const mediapipe::Packet &packet,
                                  const std::string &streamName) = 0;
    };

    class OlaGraph
    {

    public:
        OlaGraph(const mediapipe::CalculatorGraphConfig &config);
        ~OlaGraph();

        absl::Status AddCallbackHandler(std::string output_stream_name,
                                        void *callback);
        absl::Status AddMultiStreamCallbackHandler(std::vector<std::string> output_stream_names,
                                                   void *callback,
                                                   bool observe_timestamp_bounds);

        void setDelegate(std::shared_ptr<MPPGraphDelegate> delegate)
        {
            _delegate = delegate;
        }

        mediapipe::ProfilingContext *getProfiler();

        int maxFramesInFlight;

        mediapipe::CalculatorGraph::GraphInputStreamAddMode packetAddMode()
        {
            return _graph->GetGraphInputStreamAddMode();
        }

        void setPacketAddMode(mediapipe::CalculatorGraph::GraphInputStreamAddMode mode)
        {
            _graph->SetGraphInputStreamAddMode(mode);
        };

        virtual void setHeaderPacket(const mediapipe::Packet &packet, std::string streamName);

        virtual void setSidePacket(const mediapipe::Packet &packet, std::string name);

        virtual void setServicePacket(mediapipe::Packet &packet, const mediapipe::GraphServiceBase &service);

        virtual void addSidePackets(const std::map<std::string, mediapipe::Packet> &extraSidePackets);

        virtual void addFrameOutputStream(const std::string &outputStreamName,
                                          MPPPacketType packetType);

        virtual bool start();

        virtual bool sendPacket(const mediapipe::Packet &packet,
                                const std::string &streamName);

        virtual bool movePacket(mediapipe::Packet &&packet, const std::string &streamName);

        /// Sets the maximum queue size for a stream. Experimental feature, currently
        /// only supported for graph input streams. Should be called before starting the
        /// graph.
        virtual bool setMaxQueueSize(int maxQueueSize,
                                     const std::string &streamName);

#if defined(__APPLE__)
        /// Creates a MediaPipe packet wrapping the given pixelBuffer;
        mediapipe::Packet packetWithPixelBuffer(CVPixelBufferRef pixelBuffer,
                                                MPPPacketType packetType);

        /// Creates a MediaPipe packet of type Image, wrapping the given CVPixelBufferRef.
        mediapipe::Packet imagePacketWithPixelBuffer(CVPixelBufferRef pixelBuffer);

        /// Sends a pixel buffer into a graph input stream, using the specified packet
        /// type. The graph must have been started before calling this. Drops frames and
        /// returns NO if maxFramesInFlight is exceeded. If allowOverwrite is set to YES,
        /// allows MediaPipe to overwrite the packet contents on successful sending for
        /// possibly increased efficiency. Returns YES if the packet was successfully sent.
        bool sendPixelBuffer(CVPixelBufferRef imageBuffer,
                             const std::string &inputName,
                             MPPPacketType packetType,
                             const mediapipe::Timestamp &timestamp,
                             bool allowOverwrite);

        /// Sends a pixel buffer into a graph input stream, using the specified packet
        /// type. The graph must have been started before calling this. Drops frames and
        /// returns NO if maxFramesInFlight is exceeded. Returns YES if the packet was
        /// successfully sent.
        bool sendPixelBuffer(CVPixelBufferRef imageBuffer,
                             const std::string &inputName,
                             MPPPacketType packetType,
                             const mediapipe::Timestamp &timestamp);

        /// Sends a pixel buffer into a graph input stream, using the specified packet
        /// type. The graph must have been started before calling this. The timestamp is
        /// automatically incremented from the last timestamp used by this method. Drops
        /// frames and returns NO if maxFramesInFlight is exceeded. Returns YES if the
        /// packet was successfully sent.
        bool sendPixelBuffer(CVPixelBufferRef imageBuffer,
                             const std::string &inputName,
                             MPPPacketType packetType);

#endif

        /// Cancels a graph run. You must still call waitUntilDoneWithError: after this.
        void cancel();

        /// Check if the graph contains this input stream
        bool hasInputStream(const std::string &inputName);

        /// Closes an input stream.
        /// You must close all graph input streams before stopping the graph.
        /// @return YES if successful.
        bool closeInputStream(const std::string &inputName);

        /// Closes all graph input streams.
        /// @return YES if successful.
        bool closeAllInputStreams();

        /// Stops running the graph.
        /// Call this before releasing this object. All input streams must have been
        /// closed. This call does not time out, so you should not call it from the main
        /// thread.
        /// @return YES if successful.
        bool waitUntilDone();

        /// Waits for the graph to become idle.
        bool waitUntilIdle();
        
        void setUseVideoOutput(bool useVideoOutput) {
            _useVideoOutput = useVideoOutput;
        }
        
        bool useVideoOutput() {
            return _useVideoOutput;
        }

        std::weak_ptr<MPPGraphDelegate> _delegate;
        std::atomic<int32_t> _framesInFlight = 0;
        std::atomic<int32_t> _retryCount = 0;

    private:
        std::unique_ptr<mediapipe::CalculatorGraph> _graph;
        mediapipe::CalculatorGraphConfig _config;
        /// Input side packets that will be added to the graph when it is started.
        std::map<std::string, mediapipe::Packet> _inputSidePackets;
        /// Packet headers that will be added to the graph when it is started.
        std::map<std::string, mediapipe::Packet> _streamHeaders;
        /// Service packets to be added to the graph when it is started.
        std::map<const mediapipe::GraphServiceBase *, mediapipe::Packet> _servicePackets;

        /// Number of frames currently being processed by the graph.
        mediapipe::Timestamp _frameTimestamp;

        int64 _frameNumber;

        bool _started;
        bool _useVideoOutput = true;

        absl::Status performStart();

        int _maxFramesInFlight = 1;
    };

}
