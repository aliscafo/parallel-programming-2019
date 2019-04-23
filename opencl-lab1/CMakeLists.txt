cmake_minimum_required (VERSION 3.10)

project (convolution)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 11)
set (SOURCE_FILES main.cpp)

find_package (OpenCL REQUIRED)

if (${OPENCL_FOUND})
    include_directories (${OpenCL_INCLUDE_DIR})
    add_executable (convolution ${SOURCE_FILES})
    target_link_libraries (convolution LINK_PRIVATE ${OpenCL_LIBRARY})
endif()