// ========================================================================== //
// The MIT License (MIT)                                                      //
//                                                                            //
// Copyright (c) 2017 Jefferson Amstutz                                       //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// in all copies or substantial portions of the Software.                     //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
// ========================================================================== //

#include <CL/cl.hpp>

#include <iostream>

int main()
{
  std::vector<cl::Platform> all_platforms;

  cl::Platform::get(&all_platforms);

  if(all_platforms.empty()) {
    std::cout<<" No platforms found. Check OpenCL installation!\n";
    exit(1);
  }

  for (auto &p : all_platforms)
    std::cout << "Got '" << p.getInfo<CL_PLATFORM_NAME>() << "' platform\n";

  cl::Platform default_platform = all_platforms[0];

  //get default device of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.empty()) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
  }

  for (auto &d : all_devices)
    std::cout << "Got '" << d.getInfo<CL_DEVICE_NAME>() << "' device\n";

  cl::Device default_device = all_devices[0];

  std::cout << "Using device: "
            << default_device.getInfo<CL_DEVICE_NAME>() << '\n';


  cl::Context context({default_device});

  cl::Program::Sources sources;

  // kernel calculates for each element C=A+B
  std::string kernel_code =
  R"KERNEL(
    void kernel simple_add(global const int* A,
                           global const int* B,
                           global int* C) {
      C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
    }
  )KERNEL";
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);

  if (program.build({default_device}) != CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
              <<"\n";
    exit(1);
  }

  // create buffers on the device
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int)*10);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int)*10);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int)*10);

  int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);

  //write arrays A and B to the device
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*10, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*10, B);


#if 0
  //run the kernel
  cl::KernelFunctor simple_add(cl::Kernel(program,"simple_add"),
                               queue,
                               cl::NullRange,
                               cl::NDRange(10),
                               cl::NullRange);
  simple_add(buffer_A,buffer_B,buffer_C);
#endif

  cl::Kernel kernel_add(program, "simple_add");
  kernel_add.setArg(0,buffer_A);
  kernel_add.setArg(1,buffer_B);
  kernel_add.setArg(2,buffer_C);
  queue.enqueueNDRangeKernel(kernel_add,
                             cl::NullRange,
                             cl::NDRange(10),
                             cl::NullRange);
  queue.finish();

  //read result C from the device to array C
  int C[10];
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*10, C);

  std::cout<<" result: \n";

  for(int i = 0; i < 10; i++)
    std::cout<< C[i]<< " ";

  std::cout << '\n';

  return 0;
}
