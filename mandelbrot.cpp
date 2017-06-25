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

#include <cstdio>
#include <iostream>

// helper function to write the rendered image as PPM file
inline void writePPM(const std::string &fileName,
                     const int sizeX, const int sizeY,
                     const int *pixel)
{
  FILE *file = fopen(fileName.c_str(), "wb");
  fprintf(file, "P6\n%i %i\n255\n", sizeX, sizeY);
  unsigned char *out = (unsigned char *)alloca(3*sizeX);
  for (int y = 0; y < sizeY; y++) {
    const unsigned char *in = (const unsigned char *)&pixel[(sizeY-1-y)*sizeX];
    for (int x = 0; x < sizeX; x++) {
      out[3*x + 0] = in[4*x + 0];
      out[3*x + 1] = in[4*x + 1];
      out[3*x + 2] = in[4*x + 2];
    }
    fwrite(out, 3*sizeX, sizeof(char), file);
  }
  fprintf(file, "\n");
  fclose(file);
}

int main()
{
  // App setup ////////////////////////////////////////////////////////////////

  const unsigned int width  = 1200;
  const unsigned int height = 800;
  const float x0 = -2;
  const float x1 = 1;
  const float y0 = -1;
  const float y1 = 1;

  const int maxIters = 256;
  std::vector<int> buf(width*height);

  std::fill(buf.begin(), buf.end(), 0);

  // OpenCL config ////////////////////////////////////////////////////////////

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

  // OpenCL kernel construction ///////////////////////////////////////////////

  cl::Program::Sources sources;

  // kernel calculates for each element C=A+B
  std::string kernel_code =
  R"KERNEL(
    inline int mandel(const float c_re, const float c_im, const int count)
    {
      float z_re = c_re, z_im = c_im;
      int i;
      for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
          break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
      }

      return i;
    }

    void kernel simple_add(const float dx,
                           const float dy,
                           const float x0,
                           const float y0,
                           const int width,
                           const int maxIterations,
                           global int* output)
    {
      const int i = get_global_id(0); // x in image
      const int j = get_global_id(1); // y in image
      float x = x0 + i * dx;
      float y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
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

  // OpenCL kernel invocation /////////////////////////////////////////////////

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);

  // create buffers on the device
  cl::Buffer buf_cl(context, CL_MEM_READ_WRITE, sizeof(int)*width*height);

  //write arrays A and B to the device
  queue.enqueueWriteBuffer(buf_cl, CL_TRUE, 0,
                           sizeof(int)*width*height, buf.data());


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

  int arg = 0;
  kernel_add.setArg(arg++, (x1 - x0) / width);
  kernel_add.setArg(arg++, (y1 - y0) / height);
  kernel_add.setArg(arg++, x0);
  kernel_add.setArg(arg++, y0);
  kernel_add.setArg(arg++, width);
  kernel_add.setArg(arg++, maxIters);
  kernel_add.setArg(arg++, buf_cl);

  queue.enqueueNDRangeKernel(kernel_add,
                             cl::NullRange,
                             cl::NDRange(width, height),
                             cl::NullRange);
  queue.finish();

  //read result C from the device to array C
  queue.enqueueReadBuffer(buf_cl,
                          CL_TRUE,
                          0,
                          sizeof(int)*width*height,
                          buf.data());

  writePPM("mandelbrot.ppm", width, height, buf.data());

  std::cout << '\n' << "wrote output image to 'mandelbrot.ppm'" << '\n';

  return 0;
}
