## Make sure to download the memes
```
python mediapipe/examples/facial_search/images/download_images.py
```

## Run on CPU
```
bazel run --platform_suffix=_cpu \
	--copt=-fdiagnostics-color=always --run_under="cd $PWD && " \
	-c opt --define MEDIAPIPE_DISABLE_GPU=1 \
	mediapipe/examples/facial_search/desktop:facial_search \
	-- \
	--calculator_graph_config_file=mediapipe/examples/facial_search/graphs/facial_search_cpu.pbtxt \
	--images_folder_path=mediapipe/examples/facial_search/images/
```

## Run on GPU
```
bazel run --platform_suffix=_gpu \
	--copt=-fdiagnostics-color=always --run_under="cd $PWD && " \
	-c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
	mediapipe/examples/facial_search/desktop:facial_search \
	-- \
	--calculator_graph_config_file=mediapipe/examples/facial_search/graphs/facial_search_gpu.pbtxt \
	--images_folder_path=mediapipe/examples/facial_search/images/
```
