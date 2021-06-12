"""Build defs for Objectron."""

def generate_manifest_values(application_id, app_name):
    manifest_values = {
        "applicationId": application_id,
        "appName": app_name,
        "mainActivity": "com.google.mediapipe.apps.objectdetection3d.MainActivity",
        "cameraFacingFront": "False",
        "binaryGraphName": "object_detection_3d.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "2",
    }
    return manifest_values
