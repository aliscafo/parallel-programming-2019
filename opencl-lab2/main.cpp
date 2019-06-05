#define __CL_ENABLE_EXCEPTIONS

#include <OpenCL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

int n;
size_t const BLOCK_SIZE = 256;

int recursive_scan_per_block(float *output, cl::Buffer &dev_in, cl::Buffer &dev_out, cl::Program &scan_program, cl::CommandQueue &queue) {
    int depth = 1;
    while (true) {
        queue.enqueueWriteBuffer(dev_in, CL_TRUE, 0, sizeof(float) * n, output);

        cl::Kernel kernel(scan_program, "scan_hillis_steele_per_block");
        cl::KernelFunctor functor(kernel, queue, cl::NullRange, cl::NDRange(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE), 
                                                                                                          cl::NDRange(BLOCK_SIZE));
        functor(dev_in, dev_out, n, depth, cl::__local(BLOCK_SIZE * sizeof(float)), cl::__local(BLOCK_SIZE * sizeof(float)));

        queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, sizeof(float) * n, output);

        depth *= BLOCK_SIZE;

        if (depth >= n) {
            break;
        }
    }

    return depth / BLOCK_SIZE;
}

void recursive_propagation(int max_depth, float *output, cl::Buffer &dev_in, cl::Buffer &dev_out, 
                                                            cl::Program &scan_program, cl::CommandQueue &queue) {
    int depth = max_depth;

    while (true) {
        queue.enqueueWriteBuffer(dev_in, CL_TRUE, 0, sizeof(float) * n, output);

        cl::Kernel kernel(scan_program, "scan_hillis_steele_propagation");
        cl::KernelFunctor functor(kernel, queue, cl::NullRange, cl::NDRange(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE), 
                                                                                                          cl::NDRange(BLOCK_SIZE));
        functor(dev_in, dev_out, n, depth / BLOCK_SIZE);

        queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, sizeof(float) * n, output);

        depth /= BLOCK_SIZE;

        if (depth <= 1) {
            break;
        }
    }
}

int main() {
    std::freopen("input.txt", "r", stdin);
    std::freopen("output.txt", "w", stdout);

    std::cin >> n;

    float a[n];
    float output[n];

    for (int i = 0; i < n; i++) {
        std::cin >> a[i];
        output[i] = a[i];
    }

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Context context(devices);

        cl::CommandQueue queue(context, devices[0]);

        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        cl::Program scan_program(context, source);

        try {
            scan_program.build(devices);
        }
        catch (cl::Error const & e) {     
            std::string log_str = scan_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        cl::Buffer dev_in(context, CL_MEM_READ_ONLY, sizeof(float) * n);
        cl::Buffer dev_out(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        int max_depth = recursive_scan_per_block(output, dev_in, dev_out, scan_program, queue);
        recursive_propagation(max_depth, output, dev_in, dev_out, scan_program, queue);

        for (int i = 0; i < n; i++) {
            std::cout << output[i] << " ";
        }

    } catch (cl::Error &e) {
        std::cerr << std::endl << "OpenCL error: " << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}