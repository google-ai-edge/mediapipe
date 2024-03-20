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

# Dumping tensor contents to file

When using OpenVINOInferenceCalculator you can dump the input and output tensors data contents and debug info to files to verify the correctness of the data that is provided to the calculator in the graph.

To enable this functionality you need to rebuild you application with bazel cxxopt variable build 
```bash
    --cxxopt=-DOVMS_DUMP_TO_FILE=1.
```

You can pass the flag during the compilation of your application or change the value from 0 to 1 in the .bazelrc file and running 
```bash
    build_desktop_examples -b
    make run_object_detection
```

Every input and output tensor that was sent to the OpenVINOInferenceCalculator will be dumped to file in the current directory:
```bash
    ./dump/"execution_timestamp_string"/input#nr
    ./dump/"execution_timestamp_string"/output#nr
```

Example file contents for specific datatype is as follows:
```bash
    Name: normalized_input_image_tensor Shape: [1,320,320,3] Type: f32 Byte size: 1228800 Size: 307200 tensor: [ 0.403922 0.364706 0.286275 0.396078 0.356863 0.278431 0.396078 0.356863 0.278431 0.396078 0.356863 0.278431 0.396078 0.356863 0.278431 0.396078 0.356863 0.278431 0.3960 ... ]
```

The supported ov:Tensor datatypes for dumping to file are:
```bash
    TYPE_CASE(ov::element::Type_t::f64, _Float64)
    TYPE_CASE(ov::element::Type_t::f32, _Float32)
    TYPE_CASE(ov::element::Type_t::i64, int64_t)
    TYPE_CASE(ov::element::Type_t::i32, int32_t)
    TYPE_CASE(ov::element::Type_t::i16, int16_t)
    TYPE_CASE(ov::element::Type_t::i8, int8_t)
    TYPE_CASE(ov::element::Type_t::u32, uint32_t)
    TYPE_CASE(ov::element::Type_t::u16, uint16_t)
    TYPE_CASE(ov::element::Type_t::u8, uint8_t)
    TYPE_CASE(ov::element::Type_t::boolean, bool)
```

You can change the dump function implementation to log your specific datatypes by implementing it in:
```bash
    mediapipe/mediapipe/calculators/ovms/openvinoinferencedumputils.cc
```

and adding dump function in your specific calculator
```bash
    #if (OVMS_DUMP_TO_FILE == 1)
        dumpOvTensorInput(input,"input");
    #endif
```      
