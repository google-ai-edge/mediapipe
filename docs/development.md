# Development instructions
Below instructions are prepared so that you can experiment and develop your own applications with custom graphs taking advantage of OpenVINO&trade; Model Serving inference.

- run the 
```bash
make docker_build
```
 command to prepare the development and runtime environment
- start the 
```bash
docker run -it mediapipe_ovms:latest bash
```

- modify the contents of global [config_holistic.json](../mediapipe/models/ovms/config_holistic.json) or specific [object_detection config.json](../mediapipe/calculators/ovms/config.json) to change the models and graphs that are loaded for your example application
- make sure the new pbtxt and model files are present in the mediapipe/models/ovms/ directory during the application execution and are setup according to the config json file paths. Currently the existing demos setup is prepared automatically with the  command.
```bash
python setup_ovms.py --get_models'
```

- modify or create new mediapipe example target with the 
```bash
//mediapipe/graphs/object_detection:desktop_ovms_calculators
```
 and 
 ```bash
 @ovms//src:ovms_lib 
 ```
dependencies similar to [object_detection_ovms](mediapipe/examples/desktop/object_detection/BUILD) target.
- build your application graph file based on existing examples or [object_detection](mediapipe/graphs/object_detection/object_detection_desktop_ovms1_graph.pbtxt) making sure that OpenVINOModelServerSessionCalculator and OpenVINOInferenceCalculator are properly placed in the inference pipeline of your graph.
- build your new application using the bazel command inside the mediapipe_ovms container, for example in object_detection case it is:
```bash
build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_openvino'
```
- execute your application with the specific input parameters, for example in object_detection case those are:
```bash
'bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_openvino --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_openvino_graph.pbtxt --input_side_packets "input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/mediapipe/tested_video.mp4"'
```
- in object_detection example the input video will get all the objects detected by the model and return the tested_video.mp4 output.
