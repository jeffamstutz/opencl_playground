## ========================================================================== ##
## The MIT License (MIT)                                                      ##
##                                                                            ##
## Copyright (c) 2017 Jefferson Amstutz                                       ##
##                                                                            ##
## Permission is hereby granted, free of charge, to any person obtaining a    ##
## copy of this software and associated documentation files (the "Software"), ##
## to deal in the Software without restriction, including without limitation  ##
## the rights to use, copy, modify, merge, publish, distribute, sublicense,   ##
## and/or sell copies of the Software, and to permit persons to whom the      ##
## Software is furnished to do so, subject to the following conditions:       ##
##                                                                            ##
## The above copyright notice and this permission notice shall be included in ##
## in all copies or substantial portions of the Software.                     ##
##                                                                            ##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR ##
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   ##
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    ##
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER ##
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    ##
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        ##
## DEALINGS IN THE SOFTWARE.                                                  ##
## ========================================================================== ##

##############################################################
# CMake configuration
##############################################################

cmake_minimum_required(VERSION 2.8)

project(opencl_playground)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(cmake_extras)

setup_build_type()
configure_compiler()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

##############################################################
# Build test apps
##############################################################

find_package(OpenCL REQUIRED VERSION 2.2)

include_directories(
  ${CMAKE_SOURCE_DIR}
  ${OpenCL_INCLUDE_DIRS}
)

# Simple app

add_executable(simple simple.cpp)
target_link_libraries(simple ${OpenCL_LIBRARIES})

# Mandelbrot app

add_executable(mandelbrot mandelbrot.cpp)
target_link_libraries(mandelbrot ${OpenCL_LIBRARIES})
