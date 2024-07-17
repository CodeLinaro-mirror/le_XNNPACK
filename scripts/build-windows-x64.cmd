mkdir build\windows
mkdir build\windows\x64

cmd vcvarsall.bat x64

set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static
set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
@REM set CMAKE_ARGS=%CMAKE_ARGS% -G="Visual Studio 17 2022" -A=x64 -DCMAKE_CXX_FLAGS="/MP"
set CMAKE_ARGS=%CMAKE_ARGS% -G="Ninja" -DCMAKE_CXX_FLAGS="/MP"

rem Use-specified CMake arguments go last to allow overridding defaults
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo %CMAKE_ARGS%

cd build\windows\x64 && cmake ..\..\.. %CMAKE_ARGS%
cmake --build . --config Release -- -m:%NUMBER_OF_PROCESSORS%
