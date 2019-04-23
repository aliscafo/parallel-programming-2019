#define __CL_ENABLE_EXCEPTIONS

#include <OpenCL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

int main() {
  std::freopen("input.txt", "r", stdin);
  std::freopen("output.txt", "w", stdout);

  int n = 0, m = 0;
  std::cin >> n >> m;

  int a_size = n * n, b_size = m * m, c_size = n * n;
  float a[a_size], b[b_size], c[c_size];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cin >> a[i * n + j];
    }
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      std::cin >> b[i * m + j];
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      c[i * n + j] = 0;
    }
  }

  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  std::vector<cl::Kernel> kernels;

  try {
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context(devices);

    cl::CommandQueue queue(context, devices[0]);

    std::ifstream cl_file("convolution.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

    cl::Program convolution_program(context, source);

    size_t const block_size = 16;

    try {
      convolution_program.build(devices);
    }
    catch (cl::Error const & e) {     
      std::string log_str = convolution_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
      std::cout << log_str;
      return 0;
    }

    cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * a_size);
    cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * b_size);
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * c_size);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * a_size, a);
    queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * b_size, b);

    int n_mult_block_size = ((n + block_size - 1) / block_size) * block_size;
    int threads_num = n_mult_block_size * n_mult_block_size;

    cl::Kernel kernel(convolution_program, "convolution");
    cl::KernelFunctor functor(kernel, queue, cl::NullRange, cl::NDRange(threads_num), cl::NDRange(block_size));
    functor(dev_a, dev_b, dev_c, n, m);

    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * c_size, c);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        std::cout << c[i * n + j] << " ";
      }
      std::cout << "\n";
    }

  } catch (cl::Error &e) {
    std::cerr << std::endl << "OpenCL error: " << e.what() << " : " << e.err() << std::endl;

  }

  return 0;
}