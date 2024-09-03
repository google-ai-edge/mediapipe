#include "libmp_impl.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "google/protobuf/message_lite.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {

	LibMPImpl::~LibMPImpl()
	{
		LOG(INFO) << "Shutting down.";
		absl::Status status = m_graph.CloseInputStream(m_input_stream);
		if (status.ok()){
			absl::Status status1 = m_graph.WaitUntilDone();
			if (!status1.ok()){
				LOG(INFO) << "Error in WaitUntilDone(): " << status1.ToString();
			}
		} else {
			LOG(INFO) << "Error in CloseInputStream(): " << status.ToString();
		}
	}

	absl::Status LibMPImpl::Init(const char* graph, const char* inputStream){
		mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);
		MP_RETURN_IF_ERROR(m_graph.Initialize(config));
		m_input_stream.assign(inputStream);
		LOG(INFO) << "Successfully initialized LibMP graph";
		return absl::OkStatus();
	}

	bool LibMPImpl::AddOutputStream(const char* outputStream){
		m_pollers.emplace(outputStream, m_graph.AddOutputStreamPoller(outputStream));
		return m_pollers.at(outputStream).ok() ? true : false;
	}

	void LibMPImpl::SetOutputStreamMaxQueueSize(const char* outputStream, int queue_size){
		m_pollers.at(outputStream)->SetMaxQueueSize(queue_size);
	}

	bool LibMPImpl::Start(){
		const std::map<std::string, mediapipe::Packet>& extra_side_packets = {};
		bool ok = m_graph.StartRun(extra_side_packets).ok();
		LOG(INFO) << (ok ? "Started " : "Failed to start ") << "calculator graph";
		return ok;
	}

	bool LibMPImpl::Process(uint8_t* data, int width, int height, int image_format, uint64_t ts)
	{
		if (data == nullptr){
			LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
			return false;
		}
		if (!mediapipe::ImageFormat::Format_IsValid(image_format)){
			LOG(INFO) << __FUNCTION__ << " input image format (" << image_format << ") is invalid!";
			return false;
		}

		// copy input data to ImgFrame
		auto input_frame_for_input = std::make_unique<ImageFrame>();
		auto mp_image_format = static_cast<mediapipe::ImageFormat::Format>(image_format);
		input_frame_for_input->CopyPixelData(mp_image_format, width, height, data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

		m_frame_timestamp++;

		// the created ImageFrame is now owned by the input packet
		auto status = m_graph.AddPacketToInputStream(m_input_stream, mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(ts)));

		if (!status.ok()){
			LOG(INFO) << "Failed to add packet to input stream. Call m_graph.WaitUntilDone() to see error (or destroy LibMP object)";
			LOG(INFO) << "Status: " << status.ToString() << std::endl;
			return false;
		}
		return true;
	}

	bool LibMPImpl::WaitUntilIdle(){
		return m_graph.WaitUntilIdle().ok();
	}

	int LibMPImpl::GetOutputQueueSize(const char* outputStream){
		return m_pollers.at(outputStream)->QueueSize();
	}

	const void* LibMPImpl::GetOutputPacket(const char* outputStream){
		if (m_pollers.find(outputStream) == m_pollers.end()){
			LOG(INFO) << "No poller found for output stream '" << outputStream << "'. Was it created using AddOutputStream beforehand?";
			return nullptr;
		}
		auto outputPacket = std::make_unique<mediapipe::Packet>();
		if (!m_pollers.at(outputStream)->Next(outputPacket.get())){
			LOG(INFO) << "Poller for output stream '" << outputStream << "' has no next packet. Call m_graph.WaitUntilDone() to see error (or destroy LibMP object). Are models available under mediapipe/models and mediapipe/modules?";
			return nullptr;
		}
		return reinterpret_cast<const void*>(outputPacket.release());
	}

}
