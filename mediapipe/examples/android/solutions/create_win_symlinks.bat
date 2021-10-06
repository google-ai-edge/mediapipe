@rem Remove the current res dir symlinks that are for Linux and macOS and recreate res dir symlinks for Windows.
@rem This script needs administrator permission. Must run this script as administrator.

@rem for hands example app.
cd /d %~dp0
cd hands\src\main
rm res
mklink /d res ..\..\..\res

@rem for facemesh example app.
cd /d %~dp0
cd facemesh\src\main
rm res
mklink /d res ..\..\..\res

@rem for face detection example app.
cd /d %~dp0
cd facedetection\src\main
rm res
mklink /d res ..\..\..\res

dir
pause
