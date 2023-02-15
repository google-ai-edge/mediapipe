# Description

The pose_tracking_dll module allows for building a Mediapipe-based pose tracking DLL library that can be used with any C++ project. All the dependencies such as tensorflow are built statically into the dll. 

Currently, the following features are supported:
- Segmenting the person(s) of interest
- Segmenting the skeleton(s)
- Accessing the 3D coordinates of each node of the skeleton

# Prerequisites

Install Mediapipe development environment as follows. 

## Remarks

This guide assumes a Nimagna development environment. 

Otherwise, please also read the guidelines on the official Mediapipe website: https://google.github.io/mediapipe/getting_started/install.html#installing-on-windows.

## MSYS2

- Install MSYS2 from https://www.msys2.org/ (tested with version 20230127)
  - The MSYS2 installation path is referred to as `MSYS2DIR` below. 
- Install necessary packages:
  - `pacman -S git patch unzip` and confirm installation
  
## Python 3.11
  
- Install Python 3.11
  - Download Python 3.11.x Windows executable from https://www.python.org/downloads/windows/
  - Allow the installer to edit the %PATH% environment variable.  
  - Note: Newer Python version have not been tested
  - The Python installation path is referred to as `PYTHONDIR` below. 
    - Usually, this is `C:\Users\...\AppData\Local\Programs\Python\Python311` when installing only for the current user.
  - Run `pip install numpy` in a new command line.
  
## Install Visual C++ Build Tools 2019 and WinSDK

- Install Visual C++ Build Tools 2019 with WinSDK
  - Download and install Visual C++ Build Tools 2019 (16.11 used here) from https://my.visualstudio.com/Downloads?q=visual%20studio%202019%20build&wt.mc_id=o~msft~vscom~older-downloads (if link does not work, use https://visualstudio.microsoft.com/visual-cpp-build-tools/ and search for "older versions" to find the 2019 installer)
  - Install with
    - "Desktop Development with C++"
    - Individual components: select "Windows 10 SDK (10.0.19041.0)"
    ![image](https://user-images.githubusercontent.com/83065859/148920359-fc5830c2-3eb1-47d4-ba33-8b1ba783b728.png)
  - Note: VC Build Tools 2022 or newer WinSDK versions do not compile with the current code base.

## Install Bazel 5.4.0

- Download `bazel-5.4.0-windows-x86_64.exe` from https://github.com/bazelbuild/bazel/releases/tag/5.4.0 
- Put file into a folder and rename it to `bazel.exe`
- The Bazel installation path is referred to as `BAZEL_PATH` below. 

## Install OpenCV

- Download OpenCV 3.4.10 from https://sourceforge.net/projects/opencvlibrary/files/3.4.10/opencv-3.4.10-vc14_vc15.exe/download 
- Extract OpenCV into a separate folder. This folder is referred to as `OPENCVDIR` in the following steps.

## Checkout Mediapipe

- `git clone https://github.com/NimagnaAG/mediapipe`
- The repository root folder is referred to as `MEDIAPIPEDIR` below.

## Prepare Build

- Edit the `MEDIAPIPEDIR\WORKSPACE` file: 
  - Around line 215, is the "windows_opencv" repository. 
  - Adapt the path to point to `OPENCVDIR\\build` (using double backslashes):
  ```
  new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party:opencv_windows.BUILD",
    path = "OPENCVDIR\\build",
  )
  ```
- Edit the `MEDIAPIPEDIR\mediapipe\pose_tracking_dll\build.bat` file
  - Must change:
	- `BAZEL_PATH` -> path to bazel.exe (No default)
  - Build configuration
    - `MEDIAPIPE_CONFIGURATION` -> Release (Default, `opt`) or Debug (`dbg`)
  - Verify:
	- `MYSYS_PATH` -> path to `MSYS2DIR\usr\bin` (Default: `C:\msys64\usr\bin`)
	- `BAZEL_SH` -> path to bash.exe (Default: `%MYSYS_PATH%\bash.exe`)
	- `BAZEL_VS_VERSION` -> Visual Studio version (Default: `2019`)
	- `BAZEL_VC_FULL_VERSION` -> Visual Studio Build Tools full version. Depends on the version installed. See `See C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC`
	- `BAZEL_WINSDK_FULL_VERSION` -> WinSDK version (Default: `10.0.19041.0`)
	- `BAZEL_PYTHON_PATH` -> Path to python.exe (Default: `%LOCALAPPDATA_FORWARDSLASH%/Programs/Python/Python311/python.exe`)
  - Optional
	- `BAZEL_TMP_BUILD_DIR` -> A temporary build folder
	- `TARGET_PATH` -> A target folder to copy the relevant files to, i.e. the mediapipe/version folder of the external repo

# Build

- Open a Command Prompt
- `cd MEDIAPIPEDIR`
- `mediapipe\pose_tracking_dll\build.bat` ... and take a break!
- The build output can be found in the `MEDIAPIPEDIR\bazel-bin\mediapipe\pose_tracking_dll` folder.

# How to use the DLL

- Go to bazel-bin\mediapipe\pose_tracking_dll
- Link `pose_tracking_cpu.lib` and `pose_tracking_lib.dll.if.lib` statically in your project.
- Make sure `opencv_world3410.dll` and `pose_tracking_lib.dll` are accessible in your executable's DLL search path.
- Include `mediapipe\pose_tracking_dll\pose_tracking.h` header file to access the methods of the library.

# Troubleshooting

## Different OpenCV version
- If you are using a **different OpenCV version**, adapt the `OPENCV_VERSION` variable in the file `mediapipe/external/opencv_<platform>.BUILD` to the one installed in the system (https://github.com/google/mediapipe/issues/1926#issuecomment-825874197).

## Bazel issues

- If bazel fails to download packages
  - run `bazel clean --expunge` and try again.
- If bazel fails with an `fatal error C1083: Cannot open compiler generated file: '': Invalid argument`, your [path is too long](https://stackoverflow.com/questions/34074925/vs-2015-cannot-open-compiler-generated-file-invalid-argument). 
  - set `BAZEL_TMP_BUILD_DIR` in the batch file to a temporary folder with a short path
  - Note: Clean the bazel environment becomes `bazel --output_base=%BAZEL_TMP_BUILD_DIR% clean --expunge`

## Stalled build process

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
