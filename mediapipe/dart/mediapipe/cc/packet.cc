#include "google3/third_party/mediapipe/framework/packet.h";

// Copied from: https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/python/pybind/packet_creator.cc;rcl=549487063

#ifdef __cplusplus
extern "C"
{
#endif

    class PacketBinding
    {
        Packet create_string(std::string &data)
        {
            // actually call MediaPipe
        }

        Packet create_bool(bool data)
        {
        }

        Packet create_int8(int8 data)
        {
        }

        Packet create_int16(int16 data)
        {
        }

        Packet create_int32(int32 data)
        {
        }

        Packet create_int64(int64 data)
        {
        }

        Packet create_uint8(uint8 data)
        {
        }

        Packet create_uint16(uint16 data)
        {
        }

        Packet create_uint32(uint32 data)
        {
        }

        Packet create_uint64(uint64 data)
        {
        }

        Packet create_float(float data)
        {
        }

        Packet create_double(double data)
        {
        }

        Packet create_int_array(std::vector<int> &data)
        {
        }

        Packet create_float_array(std::vector<float> &data)
        {
        }

        Packet create_int_vector(std::vector<int> &data)
        {
        }

        Packet create_bool_vector(std::vector<bool> &data)
        {
        }

        Packet create_float_vector(std::vector<float> &data)
        {
        }

        Packet create_string_vector(std::vector<std::string> &data)
        {
        }

        Packet create_image_vector(std::vector<Image> &data)
        {
        }

        Packet create_packet_vector(std::vector<Packet> &data)
        {
        }

        Packet create_string_to_packet_map(std::map<std::string, Packet> &data)
        {
        }

        Packet create_matrix(Eigen::MatrixXf &matrix, bool transpose)
        {
        }

        Packet create_from_serialized(const bytes &encoding)
        {
        }
    }

#ifdef __cplusplus
}
#endif