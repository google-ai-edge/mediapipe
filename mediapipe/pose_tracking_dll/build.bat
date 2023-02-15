@echo off
setlocal ENABLEDELAYEDEXPANSION

:: ----------------------------------------------------------
:: Do not change
:: ----------------------------------------------------------

:: needs to be run from repo root
if NOT EXIST mediapipe\pose_tracking_dll\pose_tracking.cpp (
	echo Batch file must be run from repository root, i.e. mediapipe\pose_tracking_dll\build.bat
	EXIT
)
set "LOCALAPPDATA_FORWARDSLASH=%LOCALAPPDATA:\=/%"

:: ----------------------------------------------------------
:: Adapt the variables below to your local environment
:: ----------------------------------------------------------


:: path to bazel.exe
SET BAZEL_PATH=E:\repos\bazel\5.4.0

:: path to msys 
SET MYSYS_PATH=C:\msys64\usr\bin

:: path to msys bash
SET BAZEL_SH=%MYSYS_PATH%\bash.exe

:: Visual Studio C++ Build Tools
SET BAZEL_VS_VERSION=2019
SET BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\%BAZEL_VS_VERSION%\BuildTools
SET BAZEL_VC=%SBAZEL_VS%\VC

:: Visual Studio C++ Build Tools version
:: See C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC
SET BAZEL_VC_FULL_VERSION=14.29.30133

:: Windows SDK version
SET BAZEL_WINSDK_FULL_VERSION=10.0.19041.0

:: full path to python.exe
SET BAZEL_PYTHON_PATH=%LOCALAPPDATA_FORWARDSLASH%/Programs/Python/Python311/python.exe

:: Optional: temporary build path
SET BAZEL_TMP_BUILD_DIR=E:\repos\mp_output

:: Release [opt] or Debug [dbg]
SET MEDIAPIPE_CONFIGURATION=opt

:: Optional: A target path to copy the relevant files after build, i.e. the version folder of the external repo
SET TARGET_PATH=C:\Users\ChristophNiederberge\source\repos\CodeReviews\external\mediapipe\0.8.10.2_x64

:: ----------------------------------------------------------
:: Build posetracking DLL
:: ----------------------------------------------------------

IF NOT [%BAZEL_TMP_BUILD_DIR%]==[] (
    ECHO Using temporary build directory: %BAZEL_TMP_BUILD_DIR%
	bazel --output_base "%BAZEL_TMP_BUILD_DIR%" build -c %MEDIAPIPE_CONFIGURATION% --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="%BAZEL_PYTHON_PATH%" mediapipe/pose_tracking_dll:pose_tracking_cpu
) ELSE (
	bazel build -c  %MEDIAPIPE_CONFIGURATION%  --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="%BAZEL_PYTHON_PATH%" mediapipe/pose_tracking_dll:pose_tracking_cpu
)

:: ----------------------------------------------------------
:: Copy files to target folder (optional)
:: ----------------------------------------------------------
IF NOT [%TARGET_PATH%]==[] (
	IF [%MEDIAPIPE_CONFIGURATION%]==[opt] (
		ECHO Copy files to %TARGET_PATH%\x64\Release ... 
		mkdir %TARGET_PATH%\x64\Release
		copy bazel-bin\mediapipe\pose_tracking_dll\pose_tracking_lib.dll %TARGET_PATH%\x64\Release
		copy bazel-bin\mediapipe\pose_tracking_dll\pose_tracking_lib.dll.if.lib %TARGET_PATH%\x64\Release
		copy bazel-bin\mediapipe\pose_tracking_dll\opencv_world3410.dll %TARGET_PATH%\x64\Release
	) 
	IF [%MEDIAPIPE_CONFIGURATION%]==[dbg] (
		ECHO Copy files to %TARGET_PATH%\x64\Debug ... 
		mkdir %TARGET_PATH%\x64\Debug
		copy bazel-bin\mediapipe\pose_tracking_dll\pose_tracking_lib.dll %TARGET_PATH%\x64\Debug
		copy bazel-bin\mediapipe\pose_tracking_dll\pose_tracking_lib.dll.if.lib %TARGET_PATH%\x64\Debug
		copy bazel-bin\mediapipe\pose_tracking_dll\pose_tracking_lib.pdb %TARGET_PATH%\x64\Debug
		copy bazel-bin\mediapipe\pose_tracking_dll\opencv_world3410d.dll %TARGET_PATH%\x64\Debug
	) 
)