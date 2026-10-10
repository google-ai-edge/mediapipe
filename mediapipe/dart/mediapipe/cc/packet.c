#include "packet.h"

char *packet_create_string(char *data)
{

    // another possibility -
    // call MediaPipe functionality here instead of
    // having another C++ layer?

    return PacketBinding.create_string(data);
}

// psueodcode option 1 for high level C-wrapper:
// ClassificationResult classify_text(char* data) {
//     packets = create_input_packets(data)
//     results = classify(packets)
//     return results
// }

// psueodcode option 2 for high level C-wrapper:
// char* classify_text(char* data) {
//     packets = create_input_packets(data)
//     // could `results` already be a serioalized proto?
//     results = classify(packets)
//     // then we return the serialized proto straight to Dart?
//     return results
// }