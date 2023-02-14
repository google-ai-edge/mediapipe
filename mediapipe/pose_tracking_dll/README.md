## Description
The pose_tracking_dll module allows for building a Mediapipe-based pose tracking DLL library that can be used with any C++ project. All the dependencies such as tensorflow are built statically into the dll. 

Currently, the following features are supported:
- Segmenting the person(s) of interest
- Segmenting the skeleton(s)
- Accessing the 3D coordinates of each node of the skeleton

## Prerequisites

Install Mediapipe development environment as follows. 

**Note**: This guide assumes the Nimagna development environment. Otherwise, please follow the guidelines on the official Mediapipe website: https://google.github.io/mediapipe/getting_started/install.html#installing-on-windows with the important change that Bazel version 3.7.2 is required and the helpful sidemark that OpenCV version used by default in mediapipe is 3.4.10.

### Install MSYS2

- Install MSYS2 from https://www.msys2.org/ 
- Edit the %PATH% environment variable: If MSYS2 is installed to `C:\msys64`, add `C:\msys64\usr\bin` to your %PATH% environment variable.

### Install necessary packages

- Run `pacman -S git patch unzip` and confirm installation

### Install Python 3.9

- Download Python 3.9.9 Windows executable https://www.python.org/downloads/release/python-399/ and install. Note that Python 3.10 does not work.
- Allow the installer to edit the %PATH% environment variable.
- The Python installation path is referred to as `PYTHONDIR` in the following steps. Usually, this is `C:\Users\...\AppData\Local\Programs\Python\Python39` when installing only for the current user.
- Run `pip install numpy` in a new command line.

### Install Visual C++ Build Tools 2019 and WinSDK

- Download and install VC build tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/ with the following settings:
  ![image](https://user-images.githubusercontent.com/83065859/148920359-fc5830c2-3eb1-47d4-ba33-8b1ba783b728.png)

- Redistributables and WinSDK 10.0.19041.0 should be installed already. If not, install with Visual Studio or download the [WinSDK from the official Microsoft website](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/) and install.

### Install Bazel 3.7.2

- Download `bazel-3.7.2-windows-x86_64.exe` from https://github.com/bazelbuild/bazel/releases/tag/3.7.2 and rename to `bazel.exe`
- Add the location of the downloaded Bazel executable to the %PATH% environment variable. 
- Set additional Bazel environment variables as follows:
  - `BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`
  - `BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC`
  - `BAZEL_VC_FULL_VERSION=14.29.30133` - or the name of the folder you find in `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC`
  - `BAZEL_WINSDK_FULL_VERSION=10.0.19041.0`

### Checkout Mediapipe

- git clone https://github.com/NimagnaAG/mediapipe
- the repository root folder is referred to as `MEDIAPIPEDIR` in the following steps

### Install OpenCV

- Download OpenCV 3.4.10 from https://sourceforge.net/projects/opencvlibrary/files/3.4.10/opencv-3.4.10-vc14_vc15.exe/download 
- Extract OpenCV from into a folder, e.g. `C:\Users\ChristophNiederberge\source\repos\opencv_3.4.10`. This folder is referred to as `OPENCVDIR` in the following steps.
- Edit the `MEDIAPIPEDIR\WORKSPACE` file: Around line 215, is the "windows_opencv" repository. Adapt the path to point to `OPENCVDIR\\build` (using double backslashes):
  ```
  new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party:opencv_windows.BUILD",
    path = "OPENCVDIR\\build",
  )
  ```

#### Installation issue handling

- If you are using a **different OpenCV version**, adapt the `OPENCV_VERSION` variable in the file `mediapipe/external/opencv_<platform>.BUILD` to the one installed in the system (https://github.com/google/mediapipe/issues/1926#issuecomment-825874197).

## How to build
Assuming you're in `MEDIAPIPEDIR`, the root of the repository, run the following commands by replacing `PYTHONDIR` using forward slashes "/" in the path:

```
cd mediapipe
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=PYTHONDIR/python.exe pose_tracking_dll:pose_tracking_cpu
```

The results will be stored in the `bazel-bin\mediapipe\pose_tracking_dll` folder.

In the case the build stalls, pressing Ctrl+C might not be sufficient to stop the task. In that case, if you try to (resume the) build again,
the following message will be displayed:

```
Another command (pid=5300) is running. Waiting for it to complete on the server (server_pid=3684)
```

Unfortunately this process is hidden for some reason and can't be found in taskmgr. Fortunately, you can use the `taskkill` command to kill the process:

```
taskkill /F /PID 3684
```

After that, you should be able to run the build command again.

### Build debug symbols
- `dbg` can be used in place of `opt` to build the library with debug symbols in Visual Studio pdb format.

### Build issue handling

- If bazel fails to download packages, run `bazel clean --expunge` and try again.
- If bazel fails with an `fatal error C1083: Cannot open compiler generated file: '': Invalid argument`, your [path is too long](https://stackoverflow.com/questions/34074925/vs-2015-cannot-open-compiler-generated-file-invalid-argument). Actually, it is most probably the username... 
  - Adapt the call to `bazel --output_base=E:\nim_output build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=PYTHONDIR/python.exe pose_tracking_dll:pose_tracking_cpu` where `E:\nim_output build` can be replaced with some short path where bazel will store the packages and perform the build.

## How to use

- Go to bazel-bin\mediapipe\pose_tracking_dll
- Link `pose_tracking_cpu.lib` and `pose_tracking_lib.dll.if.lib` statically in your project.
- Make sure `opencv_world3410.dll` and `pose_tracking_lib.dll` are accessible in your working directory.
- Includeptyh `mediapipe\pose_tracking_dll\pose_tracking.h` header file to access the methods of the library.
