#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace py = pybind11;

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

class GraphRunner
{
public:
    GraphRunner(const std::string &graph_path)
    {
        std::string calculator_graph_config_contents;
        mediapipe::file::GetContents(
            graph_path, &calculator_graph_config_contents);
        LOG(INFO) << "Get calculator graph config contents: "
                  << calculator_graph_config_contents;
        config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

        LOG(INFO) << "Initialize the calculator graph.";
        graph.Initialize(config);

        LOG(INFO) << "Initialize the GPU.";
        std::shared_ptr<mediapipe::GpuResources> gpu_resources = mediapipe::GpuResources::Create().ValueOrDie();
        graph.SetGpuResources(std::move(gpu_resources));
        gpu_helper.InitializeForTest(graph.GetGpuResources().get());

        poller = std::make_shared<mediapipe::OutputStreamPoller>(
            graph.AddOutputStreamPoller(kOutputStream).ValueOrDie());
        graph.StartRun({});
    }

    py::array_t<unsigned char> ProcessFrame(py::array_t<unsigned char> &input)
    {
        if (input.ndim() != 3)
            throw std::runtime_error("1-channel image must be 2 dims ");
        py::buffer_info buf = input.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, mat.cols, mat.rows,
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        mat.copyTo(input_frame_mat);

        // Prepare and add graph input packet.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        gpu_helper.RunInGlContext([this, &input_frame, &frame_timestamp_us]() -> ::mediapipe::Status {
            // Convert ImageFrame to GpuBuffer.
            auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
            auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
            glFlush();
            texture.Release();
            // Send GPU image packet into the graph.
            graph.AddPacketToInputStream(
                kInputStream, mediapipe::Adopt(gpu_frame.release())
                                  .At(mediapipe::Timestamp(frame_timestamp_us)));

            return ::mediapipe::OkStatus();
        });

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet;
        if (!poller->Next(&packet))
            LOG(INFO) << "error getting packet";
        // break;
        std::unique_ptr<mediapipe::ImageFrame> output_frame;

        // Convert GpuBuffer to ImageFrame.
        gpu_helper.RunInGlContext(
            [this, &packet, &output_frame]() -> ::mediapipe::Status {
                auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
                auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
                output_frame = absl::make_unique<mediapipe::ImageFrame>(
                    mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
                    gpu_frame.width(), gpu_frame.height(),
                    mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
                gpu_helper.BindFramebuffer(texture);
                const auto info =
                    mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
                glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                             info.gl_type, output_frame->MutablePixelData());
                glFlush();
                texture.Release();
                return ::mediapipe::OkStatus();
            });

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());

        int size = output_frame_mat.channels() * output_frame_mat.size().width * output_frame_mat.size().height;
        py::array_t<unsigned char> result(size);
        py::buffer_info buf2 = result.request();

        auto pt = output_frame_mat.data;
        unsigned char *dstPt = (unsigned char *)buf2.ptr;

        for (int i = 0; i < size; i++)
        {
            dstPt[i] = pt[i];
        }

        return result;
    }

private:
    mediapipe::CalculatorGraphConfig config;
    mediapipe::CalculatorGraph graph;

    mediapipe::GlCalculatorHelper gpu_helper;

    std::shared_ptr<mediapipe::OutputStreamPoller> poller;
};

PYBIND11_MODULE(cameravtuber2, m)
{
    py::class_<GraphRunner>(m, "GraphRunner")
        .def(py::init<const std::string &>())
        .def("process_frame", &GraphRunner::ProcessFrame);
}