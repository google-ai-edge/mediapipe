#ifndef LIBMP_H
#define LIBMP_H

#include <stddef.h>
#include <stdint.h>

// Windows DLL
#if _WIN32
	#define MP_CPP_EXPORT __declspec(dllexport)  // DLL Export
	#if defined(COMPILING_DLL)
		#define MP_CPP_API __declspec(dllexport)  // DLL Export
	#else
		#define MP_CPP_API __declspec(dllimport)  // DLL Import
	#endif
#else
// Linux SO
	#define MP_CPP_EXPORT __attribute__((visibility("default")))  
	#define MP_CPP_API
#endif

namespace mediapipe {

	// MP_CPP_API applies to all functions below
	struct MP_CPP_API LibMP {
		// Create a MP graph with the specified input stream
		// Returns nullptr if initialization failed
		static LibMP* Create(const char* graph, const char* inputStream);
		virtual ~LibMP(){}

		// Create a poller for the specified output stream
		// Must be called before GetOutputPacket for the output stream
		virtual bool AddOutputStream(const char* outputStream) = 0;

		// Sets max queue size for the specified output stream
		// outputStream must have an associated poller
		// i.e., AddOutputStream must have been called beforehand with outputStream
		virtual void SetOutputStreamMaxQueueSize(const char* outputStream, int queue_size) = 0;

		// Start the MP graph
		virtual bool Start() = 0;

		// Processes one frame
		// Function copies (i.e., DOES NOT take ownership of) input data
		// Returns true if succeeded; false if failed
		virtual bool Process(uint8_t* data, int width, int height, int image_format, uint64_t ts) = 0;

		// Blocks until graph is idle
		virtual bool WaitUntilIdle() = 0;

		// Returns number of packets in queue of an outputStream
		// outputStream must have an associated poller
		// i.e., AddOutputStream must have been called beforehand with outputStream
		virtual int GetOutputQueueSize(const char* outputStream) = 0;

		// Returns the next packet available in the passed output stream as a void*
		// Returns nullptr if failed
		virtual const void* GetOutputPacket(const char* outputStream) = 0;

		// Given an output packet (passed as a void*), returns the size in bytes
		// of its contained image (if stored contiguously)
		static size_t GetOutputImageSize(const void* outputPacketVoid);

		// Copies the output image of the passed output packet to dst
		// Format is the same as that passed to Process() (ImageFormat::SRGB)
		// Returns true if succeeded; false if failed
		static bool WriteOutputImage(uint8_t* dst, const void* outputPacketVoid);

		// Returns true if packet is empty, false otherwise
		static bool PacketIsEmpty(const void* outputPacketVoid);

		// Get an output packet's underlying protobuf message
		// Returns nullptr if failed
		static const void* GetPacketProtoMsg(const void* outputPacketVoid);

		// Get an output packet's underlying protobuf message at index idx (packet must be a vector of messages)
		// Returns nullptr if failed
		static const void* GetPacketProtoMsgAt(const void* outputPacketVoid, unsigned int idx);

		// Get # elements in an output packet's protobuf message vec
		static size_t GetPacketProtoMsgVecSize(const void* outputPacketVoid);

		// Get size (in bytes) of a single output protobuf message
		static size_t GetProtoMsgByteSize(const void* outputProtoVoid);

		// Write serialized form of a protobuf message to passed byte array dst
		// Returns true if succeeded; false if failed
		static bool WriteProtoMsgData(uint8_t* dst, const void* outputProtoVoid, int size);

		// Deletes packet
		static void DeletePacket(const void* packetVoid);
	};

}
#endif // LIBMP_H
