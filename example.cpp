/*
 * This file is part of QCL, a small OpenCL interface which makes it quick and
 * easy to use OpenCL.
 *
 * Copyright (c) 2016,2017, Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vector>
#include <iostream>
#include "qcl.hpp"

// Here, we define our OpenCL code.
// While QCL can also just use source files or source
// strings, QCL modules are more powerful and very easy to use. (see
// the documentation in qcl_module.hpp for more information).
// Think of a QCL module as a unit of CL code which can optionally contain
// one or several kernels.
// In this example, we will define an OpenCL kernel, which adds values
// of arbitrary type, using the feature of QCL modules to import
// template arguments.
// QCL module kernels are made accessible using QCL_ENTRYPOINT(),
// and can then be called directly, similar to a normal function.
// These kernels are automatically compiled on their first call.
template<class T>
QCL_STANDALONE_MODULE(test_module)
QCL_ENTRYPOINT(add)
QCL_STANDALONE_SOURCE
(
  QCL_IMPORT_TYPE(T)
  QCL_RAW
  (
    __kernel void add(__global T* input_a,
                      __global T* input_b,
                      __global T* output)
    {
      int gid = get_global_id(0);
      output[gid] = input_a[gid] + input_b[gid];
    }
  )
);

template<class T>
void test_kernel(const qcl::device_context_ptr& ctx);

int main()
{
  // Create environment. The environment deals with different OpenCL platforms.
  qcl::environment env;

  // Create global context. A global context allows to create contexts
  // for several devices.
  // In this example, we will create a global context containing all devices
  // from one OpenCL platform.
  // get_platform_by_preference() returns the platform that best matches the keyword
  // in the arguments. The first arguments will be preferred.
  cl::Platform platform = env.get_platform_by_preference({"NVIDIA", "AMD", "Intel"});
  qcl::global_context_ptr global_ctx = env.create_global_context(platform);
  // Alternatively, you could e.g. also create a global context spanning several platforms.
  // E.g., create one global context containing all GPUs:
  // qcl::global_context_ptr global_ctx = env.create_global_gpu_context();

  if(global_ctx->get_num_devices() == 0)
  {
    std::cout << "No OpenCL devices available!" << std::endl;
    return -1;
  }

  // In this example, we will simply use the first device.
  // A device_context contains all we need to execute OpenCL code:
  // command queues, caches for compiled programs and kernels and so on.
  qcl::device_context_ptr ctx = global_ctx->device(0);

  std::cout << "Using device: " << ctx->get_device_name() << std::endl;

  // QCL correctly distinguishes kernels in modules with different
  // template arguments:
  // Create test data and execute our kernel for int and float.
  // The problem is implemented such that for int,
  // we effectively calculate 2*i, and 2*(i+0.3) for float.
  test_kernel<int>(ctx);
  test_kernel<float>(ctx);
}

// This function creates test data and executes
// our kernel. The generated test data is
// input_a[i] = input_b[i] = i for integer types and
// input_a[i] = input_b[i] = i+0.3 for float types.
template<class T>
void test_kernel(const qcl::device_context_ptr& ctx)
{
  // Create input and output buffers, fill with data
  std::vector<T> host_input(64);
  for(std::size_t i = 0; i < host_input.size(); ++i)
    host_input[i] = static_cast<T>(i)+static_cast<T>(0.3);

  cl::Buffer input_a, input_b, output;
  ctx->create_buffer<T>(input_a, host_input.size());
  ctx->create_buffer<T>(input_b, host_input.size());
  ctx->create_buffer<T>(output, host_input.size());

  ctx->memcpy_h2d(input_a, host_input.data(), host_input.size());
  ctx->memcpy_h2d(input_b, host_input.data(), host_input.size());

  // Kernel calls are easy! We can directly call a function for the kernel,
  // specifiy the launch parameters, and pass the kernel arguments
  // like arguments to a normal function.
  cl::NDRange global_size{host_input.size()};
  cl::NDRange group_size{16};
  cl_int error = test_module<T>::add(ctx, global_size, group_size)(input_a, input_b, output);
  qcl::check_cl_error(error, "Could not enqueue kernel!");

  ctx->get_command_queue().finish();

  std::vector<T> host_output(host_input.size());
  ctx->memcpy_d2h(host_output.data(), output, host_output.size());

  for(std::size_t i = 0; i < host_output.size(); ++i)
    std::cout << "output[" << i << "]=" << host_output[i] << std::endl;
}

