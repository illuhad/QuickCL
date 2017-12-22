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

#ifndef QCL_HPP
#define QCL_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120


#include <CL/cl2.hpp>

// Disable OpenCL<->OpenGL interoperability by default
#ifndef WITH_GL_INTEROP
#define WITHOUT_GL_INTEROP
#endif


#ifndef WITHOUT_GL_INTEROP
#ifdef WIN32
// What is the correct header?
#elif __APPLE__
// What is the correct header?
#else
#include <GL/glew.h>
#include <GL/glx.h>
#endif

#endif

#include <sstream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>
#include <cassert>


namespace qcl {

namespace detail {
/// Some functions of the OpenCL C++ Wrappers
/// will include a trailing \0 in the strings,
/// which can lead to problems. This function removes
/// '\0' from the string
static void remove_zeros(std::string& s)
{
  std::size_t pos = std::string::npos;
  while((pos = s.find('\0')) != std::string::npos)
    s.erase(pos, 1);
}

}

/// Class representing OpenCL errors that are encountered by QCL.
class qcl_error : public std::runtime_error
{
public:
  /// \param what A description of the error
  /// \param opencl_error The encountered error code
  explicit qcl_error(const std::string& what, cl_int opencl_error)
    : runtime_error{what},
      _cl_err{opencl_error}
  {}

  qcl_error(const qcl_error& other) = default;
  qcl_error& operator=(const qcl_error& other) = default;

  virtual ~qcl_error(){}

  /// \return The error code of the OpenCL error
  cl_int get_opencl_error_code() const noexcept
  {
    return _cl_err;
  }
private:
  cl_int _cl_err;
};

/// Simple error checking function. In case of an error, throws
/// \c qcl_error
/// \throws \c std::rutime_error
/// \param err The error code that shall be checked
/// \param msg The error message
static
void check_cl_error(cl_int err, const std::string& msg)
{
  if(err != CL_SUCCESS)
  {
    std::stringstream sstr;
    sstr << "OpenCL error " << err << ": " << msg;
    throw qcl_error{sstr.str(), err};
  }
}

using kernel_ptr = std::shared_ptr<cl::Kernel>;
using buffer_ptr = std::shared_ptr<cl::Buffer>;

using command_queue_id = std::size_t;


/// This class simplifies passing arguments to kernels
/// by counting the argument index automatically.
class kernel_argument_list
{
public:
  /// Construct object
  /// \param kernel The kernel for which the arguments shall
  /// be set
  explicit kernel_argument_list(const kernel_ptr& kernel)
    : _kernel(kernel), _num_arguments()
  {
    assert(kernel != nullptr);
  }

  /// Pass argument to the kernel.
  /// \return The OpenCL error code
  /// \param data The kernel argument
  template<class T>
  cl_int push(const T& data)
  {
    cl_int err = _kernel->setArg(_num_arguments, data);

    ++_num_arguments;
    return err;
  }

  /// Pass argument to the kernel.
  /// \return The OpenCL error code
  /// \param data The kernel argument
  /// \param size The size in bytes of the argument
  cl_int push(const void* data, std::size_t size)
  {
    cl_int err = _kernel->setArg(_num_arguments, size, data);

    ++_num_arguments;
    return err;
  }

  /// \return The number of arguments that have been set
  unsigned get_num_pushed_arguments() const noexcept
  {
    return _num_arguments;
  }

  /// Resets the argument counter, hence allowing to set
  /// kernel arguments again.
  void reset()
  {
    _num_arguments = 0;
  }

private:
  kernel_ptr _kernel;
  unsigned _num_arguments;
};




/// Represents the OpenCL context of a device. This class contains everything
/// that is needed to execute OpenCL commands on a device. It stores one cl::Context,
/// at least one cl::CommandQueue and a number of kernels that have been compiled for the device.
class device_context
{
public:
  /// Create NULL object
  device_context()
  {}
  
  /// \param platform The OpenCL platform to which the device belongs
  /// \param device The OpenCL device to which the instance shall be bound
  device_context(const cl::Platform& platform, const cl::Device& device)
  : _device(device)
  {
    cl_int err;
    cl_context_properties cprops[3] = 
      {CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
     
    _context = cl::Context(
       device, 
       cprops,
       NULL,
       NULL,
       &err);
    
    check_cl_error(err, "Could not spawn CL context!");
    
    init_device();
  }
  
  /// Create object from existing cl context and a given device
  device_context(const cl::Context& context, const cl::Device& device)
  : _context(context), _device(device)
  {
    init_device();
  }
  
  device_context(const device_context& other) = delete;
  device_context& operator=(const device_context& other) = delete;
  
  /// \return The device that is accessed by this context
  const cl::Device& get_device() const
  {
    return _device;
  }
  
  /// \return The underlying OpenCL context
  const cl::Context& get_context() const
  {
    return _context;
  }
  
  /// \return The underlying OpenCL command queue
  /// \param queue The id of the command queue. The id must be greater or
  /// equal than 0 and smaller than \c get_num_command_queues().
  const cl::CommandQueue& get_command_queue(command_queue_id queue) const
  {
    assert(queue < _queues.size());
    return _queues[queue];
  }

  /// \return The underlying OpenCL command queue
  /// \param queue The id of the command queue. The id must be greater or
  /// equal than 0 and smaller than \c get_num_command_queues().
  cl::CommandQueue& get_command_queue(command_queue_id queue)
  {
    assert(queue < _queues.size());
    return _queues[queue];
  }

  /// \return The underlying OpenCL command queue
  const cl::CommandQueue& get_command_queue() const
  {
    return _queues[0];
  }

  /// \return The underlying OpenCL command queue
  cl::CommandQueue& get_command_queue()
  {
    return _queues[0];
  }



  /// \return The device name
  std::string get_device_name() const
  {
    std::string dev_name;
    check_cl_error(_device.getInfo(CL_DEVICE_NAME, &dev_name), 
                  "Could not obtain device information!");
    
    // Apparently, the OpenCL wrappers leave '\0' in the string (Bug?). This 
    // can lead to problems when concatenating the device_name with other strings
    // and using c_str()
    detail::remove_zeros(dev_name);
    return dev_name;
  }

  /// \return The device vendor
  std::string get_device_vendor() const
  {
    std::string vendor;
    check_cl_error(_device.getInfo(CL_DEVICE_VENDOR, &vendor),
                   "Could not obtain device information!");

    detail::remove_zeros(vendor);
    return vendor;
  }

  /// \return The OpenCL version supported by the device
  std::string get_device_cl_version() const
  {
    std::string cl_version;
    check_cl_error(_device.getInfo(CL_DEVICE_VERSION, &cl_version),
                   "Could not obtain device information!");

    detail::remove_zeros(cl_version);
    return cl_version;

  }

  /// \return The driver version
  std::string get_driver_version() const
  {
    std::string driver_version;
    check_cl_error(_device.getInfo(CL_DRIVER_VERSION, &driver_version),
                   "Could not obtain device information!");

    detail::remove_zeros(driver_version);
    return driver_version;
  }
  
  /// \return The type of the device
  cl_device_type get_device_type() const
  {
    return _device_type;
  }
  
  /// \return Whether the device is classified as a CPU
  bool is_cpu_device() const
  {
    return get_device_type() == CL_DEVICE_TYPE_CPU;
  }
  
  /// \return Whether the device is classified as a GPU
  bool is_gpu_device() const
  {
    return get_device_type() == CL_DEVICE_TYPE_GPU;
  }
  
  /// Compiles a file containing CL source code, and creates kernel objects
  /// for the kernels contained in the file.
  /// \param cl_source_file The CL source file
  /// \param kernel_names The names of the kernels in the file as strings
  /// \param scope The scope under which the kernels will be registered. If
  /// scope is not an empty string, kernels will be registered under the name
  /// \c scope::kernel_name
  void register_source_file(const std::string& cl_source_file, 
                            const std::vector<std::string>& kernel_names,
                            const std::string& scope = "")
  {
    std::string source_code = this->read_source_file(cl_source_file);

    register_source_code(source_code, kernel_names, cl_source_file, scope);
  }
  
  /// Compiles CL source code in the form of a \c std::string, and creates kernel objects
  /// for the kernels defined in the string.
  /// \param cl_source The CL source code
  /// \param kernel_names The names of the kernels defined in the string
  void register_source_code(const std::string& cl_source,
                            const std::vector<std::string>& kernel_names)
  {
    // generate program name
    std::string program_name;
    for(const std::string& str : kernel_names)
      program_name += str;

    register_source_code(cl_source, kernel_names, program_name);
  }

  /// Compiles CL source code in the form of a \c std::string, and creates kernel objects
  /// for the kernels defined in the string.
  /// \param source_code The CL source code
  /// \param kernel_names The names of the kernels defined in the string
  /// \param program_name A unique identifier of the source code. This
  /// identifier will be used to cache the compiled program.
  /// \param scope The scope under which the kernels will be registered. If
  /// scope is not an empty string, kernels will be registered under the name
  /// \c scope::kernel_name
  void register_source_code(const std::string& source_code,
                            const std::vector<std::string>& kernel_names,
                            const std::string& program_name,
                            const std::string& scope = "")
  {
    std::string scope_prefix = "";
    if(scope.size() > 0)
      scope_prefix = scope + "::";

    std::vector<std::string> new_kernels;

    for(const auto& kernel_name : kernel_names)
    {
      if(_kernels.find(scope_prefix+kernel_name) == _kernels.end())
        new_kernels.push_back(kernel_name);
    }

    if(new_kernels.size() > 0)
    {
      auto cached_program = _program_cache.find(program_name);
      if(cached_program == _program_cache.end())
      {
        cl::Program prog;
        compile_source(source_code, prog);
        _program_cache[program_name] = prog;
      }
    
      load_kernels(_program_cache[program_name], new_kernels, scope);
    }
  }
  
  /// Compiles a CL source module and creates kernel objects
  /// \tparam Source_module_type The source module
  /// \param kernel_names The names of the kernels defined in the source module
  template<class Source_module_type>
  void register_source_module(const std::vector<std::string>& kernel_names)
  {
    register_source_code(Source_module_type::_qcl_source(),
                         kernel_names,
                         Source_module_type::_qcl_get_module_name(),
                         Source_module_type::_qcl_get_module_name());
  }
  
  /// \return A kernel object
  /// \param kernel_name The name of the kernel. Throws \c std::runtime_error if
  /// the kernel is not found.
  kernel_ptr get_kernel(const std::string& kernel_name)
  {
    kernel_ptr kernel = _kernels[kernel_name];
    
    if(!kernel)
      throw std::runtime_error("Requested kernel could not be found!");

    return kernel;
  }
  
  /// Create an OpenCL buffer object. For CPU devices, it will
  /// be attempted to construct a zero-copy buffer. In this case
  /// \c initial_data must remain valid!
  /// \return A pointer to the created buffer object
  /// \param flags The OpenCL flags for the allocated memory
  /// \param size The number of elements to allocate
  /// \tparam T The data type for which memory should be allocated
  /// \param initial_data A pointer to data that will be used to initialize the buffer
  /// if \c initial_data is not \c NULL.
  template<class T>
  buffer_ptr create_buffer(cl_mem_flags flags,
                   std::size_t size,
                   T* initial_data = nullptr) const
  {
    if(this->is_cpu_device())
    {
      // Try a zero-copy buffer
      if(!initial_data)
        flags |= CL_MEM_ALLOC_HOST_PTR;
      else
        flags |= CL_MEM_USE_HOST_PTR;
    }
    else
    {
      if(initial_data)
        flags |= CL_MEM_COPY_HOST_PTR;
    }

    cl_int err;
    buffer_ptr buff = buffer_ptr(new cl::Buffer(_context, flags, size * sizeof(T), initial_data, &err));
    check_cl_error(err, "Could not create buffer object!");
    
    return buff;
  }
  
  /// Create an OpenCL buffer object. For CPU devices, it will
  /// be attempted to construct a zero-copy buffer. In this case
  /// \c initial_data must remain valid!
  /// \param out The resulting new buffer object
  /// \param flags The OpenCL flags for the allocated memory
  /// \param size The number of elements to allocate
  /// \tparam T The data type for which memory should be allocated
  /// \param initial_data A pointer to data that will be used to initialize the buffer
  /// if \c initial_data is not \c NULL.
  template<class T>
  void create_buffer(cl::Buffer& out,
                   cl_mem_flags flags,
                   std::size_t size,
                   T* initial_data = nullptr) const
  {
    if(this->is_cpu_device())
    {
      // Try a zero-copy buffer
      if(!initial_data)
        flags |= CL_MEM_ALLOC_HOST_PTR;
      else
        flags |= CL_MEM_USE_HOST_PTR;
    }
    else
    {
      if(initial_data)
        flags |= CL_MEM_COPY_HOST_PTR;
    }
    
    cl_int err;
    out = cl::Buffer(_context, flags, size * sizeof(T), initial_data, &err);
    check_cl_error(err, "Could not create buffer object!");
  }

  /// Create a read-write OpenCL buffer object. For CPU devices, it will
  /// be attempted to construct a zero-copy buffer. In this case
  /// \c initial_data must remain valid!
  /// \param out The created buffer
  /// \param size The number of elements of the buffer
  /// \param initial_data The initial_data buffer.
  template<class T>
  void create_buffer(cl::Buffer& out,
                     std::size_t size,
                     T* initial_data = nullptr) const
  {
    this->create_buffer<T>(out, CL_MEM_READ_WRITE, size, initial_data);
  }

  /// Create a read-write OpenCL buffer object. For CPU devices, it will
  /// be attempted to construct a zero-copy buffer. In this case
  /// \c initial_data must remain valid!
  /// \param out The created buffer
  /// \param size The number of elements of the buffer
  /// \param initial_data The initial_data buffer.
  template<class T>
  buffer_ptr create_buffer(
                     std::size_t size,
                     T* initial_data = nullptr) const
  {
    return create_buffer<T>(CL_MEM_READ_WRITE, size, initial_data);
  }

  /// Creates a buffer that is read-only for the compute device.
  /// \return A pointer to the created buffer object
  /// \param size The number of elements to allocate
  /// \tparam T The data type for which memory should be allocated
  /// \param initial_data A pointer to data that will be used to initialize the buffer
  /// if \c initial_data is not \c NULL.
  template<class T>
  buffer_ptr create_input_buffer(
                   std::size_t size,
                   T* initial_data = nullptr) const
  {
    return create_buffer<T>(CL_MEM_READ_ONLY, size, initial_data);
  }
  
  /// Creates a buffer that is write-only for the compute device.
  /// \return A pointer to the created buffer object
  /// \param size The number of elements to allocate
  /// \tparam T The data type for which memory should be allocated
  /// \param initial_data A pointer to data that will be used to initialize the buffer
  /// if \c initial_data is not \c NULL.
  template<class T>
  buffer_ptr create_output_buffer(
                   std::size_t size,
                   T* initial_data = nullptr) const
  {
    return create_buffer<T>(CL_MEM_WRITE_ONLY, size, initial_data);
  }
  
  /// Creates a buffer that is read-only for the compute device.
  /// \param out The created buffer object
  /// \param size The number of elements to allocate
  /// \tparam T The data type for which memory should be allocated
  /// \param initial_data A pointer to data that will be used to initialize the buffer
  /// if \c initial_data is not \c NULL.
  template<class T>
  void create_input_buffer(
                   cl::Buffer& out,
                   std::size_t size,
                   T* initial_data = nullptr) const
  { 
    create_buffer<T>(out, CL_MEM_READ_ONLY, size, initial_data);
  }
  
  /// Creates a buffer that is write-only for the compute device.
  /// \param out The created buffer object
  /// \param size The number of elements to allocate
  /// \tparam T The data type for which memory should be allocated
  /// \param initial_data A pointer to data that will be used to initialize the buffer
  /// if \c initial_data is not \c NULL.
  template<class T>
  void create_output_buffer(
                   cl::Buffer& out,
                   std::size_t size,
                   T* initial_data = nullptr) const
  {
    create_buffer<T>(out, CL_MEM_WRITE_ONLY, size, initial_data);
  }
  
  template<class T>
  void memcpy_h2d(const cl::Buffer& buff,
                  const T* data,
                  std::size_t size,
                  command_queue_id queue = 0) const
  {   
    cl_int err;

    err = get_command_queue(queue).enqueueWriteBuffer(buff, CL_TRUE,
                                                      0, size * sizeof(T), data);

    check_cl_error(err, "Could not enqueue buffer write!");
  }

  template<class T>
  void memcpy_h2d_async(const cl::Buffer& buff,
                  const T* data, std::size_t size, cl::Event* event, 
                  const std::vector<cl::Event>* dependencies=nullptr,
                  command_queue_id queue = 0) const
  {
    cl_int err;
    err = get_command_queue(queue).enqueueWriteBuffer(buff,
                                                      CL_FALSE, 0, size * sizeof(T),
                                                      data, dependencies, event);

    check_cl_error(err, "Could not enqueue async buffer write!");
  }
  
  template<class T>
  void memcpy_d2h(T* data, 
                  const cl::Buffer& buff,
                  std::size_t size,
                  command_queue_id queue = 0) const
  {
    cl_int err;

    err = get_command_queue(queue).enqueueReadBuffer(buff,
                                                     CL_TRUE, 0, size * sizeof(T), 
                                                     data);

    check_cl_error(err, "Could not enqueue buffer write!");
  }
  
  template<class T>
  void memcpy_d2h_async(T* data, 
                  const cl::Buffer& buff,
                  std::size_t size,
                  cl::Event* event, 
                  const std::vector<cl::Event>* dependencies = nullptr,
                  command_queue_id queue = 0) const
  {
    cl_int err;
    err = get_command_queue(queue).enqueueReadBuffer(buff, CL_FALSE, 0, size * sizeof(T),
                                                     data, dependencies, event);

    check_cl_error(err, "Could not enqueue async buffer write!");
  }


  template<class T>
  void memcpy_h2d(const cl::Buffer& buff,
                  const T* data, 
                  std::size_t begin, std::size_t end,
                  command_queue_id queue = 0) const
  {
    assert(end > begin);
    std::size_t size = end - begin;

    cl_int err;
    err = get_command_queue(queue).enqueueWriteBuffer(buff, CL_TRUE,
                                                      begin * sizeof(T),
                                                      size * sizeof(T),
                                                      data);
    check_cl_error(err, "Could not enqueue buffer write!");
  }

  template<class T>
  void memcpy_h2d_async(const cl::Buffer& buff,
                  const T* data, 
                  std::size_t begin, 
                  std::size_t end, 
                  cl::Event* event, 
                  const std::vector<cl::Event>* dependencies=nullptr,
                  command_queue_id queue = 0) const
  {
    assert(end > begin);
    std::size_t size = end - begin;

    cl_int err;
    err = get_command_queue(queue).enqueueWriteBuffer(buff, CL_FALSE,
                                                      begin * sizeof(T), 
                                                      size * sizeof(T),
                                                      data, dependencies, event);

    check_cl_error(err, "Could not enqueue async buffer write!");
  }
  
  template<class T>
  void memcpy_d2h(T* data, 
                  const cl::Buffer& buff,
                  std::size_t begin,
                  std::size_t end,
                  command_queue_id queue = 0) const
  {
    assert(end > begin);
    std::size_t size = end - begin;

    cl_int err;
    err = get_command_queue(queue).enqueueReadBuffer(buff, CL_TRUE,
                                                     begin * sizeof(T), 
                                                     size * sizeof(T),
                                                     data);

    check_cl_error(err, "Could not enqueue buffer write!");
  }
  
  template<class T>
  void memcpy_d2h_async(T* data, 
                  const cl::Buffer& buff,
                  std::size_t begin,
                  std::size_t end,
                  cl::Event* event, 
                  const std::vector<cl::Event>* dependencies = nullptr,
                  command_queue_id queue = 0) const
  {
    assert(end > begin);
    std::size_t size = end - begin;

    cl_int err;
    err = get_command_queue(queue).enqueueReadBuffer(buff, CL_FALSE,
                                                     begin * sizeof(T),
                                                     size * sizeof(T),
                                                     data, dependencies, event);

    check_cl_error(err, "Could not enqueue async buffer write!");
  }
  
  /// Queries the OpenCL extensions supported by a given device.
  /// \param device The OpenCL device
  /// \param extensions A string that will be used to store a list of all supported extensions
  static void get_supported_extensions(const cl::Device& device, std::string& extensions)
  {
    check_cl_error(device.getInfo(CL_DEVICE_EXTENSIONS, &extensions),
                   "Could not query extensions!");
  }
  
  /// \return Whether a given OpenCL extension is supported by a given device
  /// \param device The OpenCL device
  /// \param extension The name of the extension
  static bool is_extension_supported(const cl::Device& device, const std::string& extension)
  {
    std::string extensions;
    get_supported_extensions(device, extensions);
    return extensions.find(extension) != std::string::npos;
  }
  
  /// Queries the extensions supported by the device.
  /// \param extensions A string that will be used to store a list of the supported extensions
  void get_supported_extensions(std::string& extensions) const
  {
    get_supported_extensions(this->_device, extensions);
  }
  
  /// \return Whether a given OpenCL extension is supported by the device
  /// \param extension The name of the extension
  bool is_extension_supported(const std::string& extension) const
  {
    return is_extension_supported(this->_device, extension);
  }

  /// Adds a new command queue to the device. Note that one command queue
  /// is always created during the constuction of the \c device_context
  /// object.
  /// \return The id of the created queue
  /// \param props Optional properties of the queue
  command_queue_id add_command_queue(cl_command_queue_properties props = 0)
  {
    cl_int err;
    _queues.push_back(cl::CommandQueue(_context, _device, props, &err));
    
    check_cl_error(err, "Could not create command queue!");

    return _queues.size() - 1;
  }

  /// Adds a new out-of-order command queue to the device
  /// \return The id of the created queue
  command_queue_id add_out_of_order_command_queue()
  {
    return add_command_queue(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  }

  /// \return The number of command queues
  std::size_t get_num_command_queues() const
  {
    return _queues.size();
  }

  /// Ensures that at least \c num_queues are available.
  /// If less command queues are available, creates new command
  /// queues until enough are available.
  void require_several_command_queues(std::size_t num_queues)
  {
    while(get_num_command_queues() < num_queues)
      add_command_queue();
  }

  cl_int enqueue_ndrange_kernel(const kernel_ptr& kernel,
                                const cl::NDRange& minimum_num_work_items,
                                const cl::NDRange& num_local_items,
                                cl::Event* event = nullptr,
                                const cl::NDRange& offset = cl::NullRange,
                                const std::vector<cl::Event>* dependencies = nullptr,
                                command_queue_id queue = 0)
  {
    assert(queue < get_num_command_queues());
    assert(minimum_num_work_items.dimensions() == num_local_items.dimensions());

    cl::NDRange global = minimum_num_work_items;
    for(std::size_t i = 0; i < minimum_num_work_items.dimensions(); ++i)
    {
      std::size_t work_items = minimum_num_work_items.get()[i];
      std::size_t local_items = num_local_items.get()[i];

      std::size_t multiple = (work_items/local_items)*local_items;
      if(multiple != work_items)
        multiple += local_items;

      assert(multiple % local_items == 0 && multiple >= work_items);
      global.get()[i] = multiple;
    }

    cl_int err = get_command_queue(queue).enqueueNDRangeKernel(*kernel,
                                                               offset,
                                                               global,
                                                               num_local_items,
                                                               dependencies,
                                                               event);
    return err;
  }



private:

  /// Loads and creates specified kernel objects
  /// \param prog The compiled OpenCL program
  /// \param kernel_names The names of the kernels that shall be loaded
  /// \param scope The scope under which the kernel_names should be cached.
  /// If scope is not an empty string, scope:: will be used as prefix
  /// for all kernel names.
  void load_kernels(const cl::Program& prog, 
                    const std::vector<std::string>& kernel_names,
                    const std::string& scope)
  {
    std::string prefix=scope;
    if(scope.size() > 0)
      prefix += "::";

    for(std::size_t i = 0; i < kernel_names.size(); ++i)
    {
      cl_int err;
      kernel_ptr kernel = kernel_ptr(new cl::Kernel(prog, kernel_names[i].c_str(), &err));
      check_cl_error(err, "Could not create kernel object!");

      _kernels[prefix+kernel_names[i]] = kernel;
    }
  }
  
  /// Initializes the device and creates a command queue.
  void init_device()
  {
    add_command_queue();

    check_cl_error(_device.getInfo(CL_DEVICE_TYPE, &_device_type),
                   "get_device_type(): Could not obtain device type");
  }
  
  /// Compiles OpenCL source code and creates a cl::Program object.
  /// \param program_src The OpenCL source code
  /// \param program The newly created cl::Program object.
  void compile_source(const std::string& program_src,
                      cl::Program& program) const
  {
    cl::Program::Sources src{1, 
                            program_src};

    program = cl::Program(_context, src);

    cl_int err = program.build(std::vector<cl::Device>(1,_device), "");

    if(err != CL_SUCCESS)
    {
      std::string log;
      program.getBuildInfo(_device, CL_PROGRAM_BUILD_LOG, &log);

      std::stringstream sstr;
      sstr << get_device_name() << ": Could not compile CL source: " << log; 
      std::string err_msg = sstr.str();

      throw std::runtime_error(err_msg);
    }
  }
  
  /// Read source code from a file
  /// \param filename The name of the file
  /// \return The source code
  std::string read_source_file(const std::string& filename) const
  {
    std::ifstream file(filename.c_str());
    if(!file.is_open())
      throw std::runtime_error("Could not open CL source file!");


    std::string program_src = std::string(std::istreambuf_iterator<char>(file),
                                          std::istreambuf_iterator<char>());
    
    return program_src;
  }
  
  /// The OpenCL context for this device
  cl::Context _context;

  /// The device object
  cl::Device _device;
  
  /// The command queues for this device. One command
  /// queue is always created during initialization.
  std::vector<cl::CommandQueue> _queues;
  
  /// Stores the names of the kernels and their
  /// corresponding kernel objects.
  std::map<std::string, kernel_ptr> _kernels;
  /// Caches compiled programs so that they do not
  /// need to be compiled again if a different kernel
  /// from the same program is required
  std::map<std::string, cl::Program> _program_cache;
  
  /// The type of this device
  cl_device_type _device_type;
};

using device_context_ptr = std::shared_ptr<device_context>;
using const_device_context_ptr = std::shared_ptr<const device_context>;



template<class T>
class local_memory
{
public:
  local_memory(std::size_t num_elements)
    : _num_elements{num_elements}
  {}

  std::size_t get_num_elements() const
  {
    return _num_elements;
  }

  std::size_t get_size() const
  {
    return _num_elements * sizeof(T);
  }
private:
  std::size_t _num_elements;
};

template<class T>
class raw_memory
{
public:
  raw_memory(T* data, std::size_t num_bytes)
    : _data{data}, _num_bytes{num_bytes}
  {}

  T* get_data() const
  {
    return _data;
  }

  std::size_t get_size() const
  {
    return _num_bytes;
  }
private:
  T* _data;
  std::size_t _num_bytes;
};

class kernel_call
{
public:
  kernel_call(const qcl::device_context_ptr& ctx,
              const qcl::kernel_ptr& kernel,
              const cl::NDRange& minimum_work_dim,
              const cl::NDRange& group_dim,
              cl::Event* evt = nullptr,
              std::vector<cl::Event>* dependencies = nullptr)
    : _ctx{ctx},
      _kernel{kernel},
      _args{kernel},
      _work_dim{minimum_work_dim},
      _group_dim{group_dim},
      _evt{evt},
      _dependencies{dependencies}
  {
  }

  void set_event(cl::Event* evt)
  {
    this->_evt = evt;
  }

  void set_dependencies(std::vector<cl::Event>* dependencies)
  {
    this->_dependencies = dependencies;
  }

  template<typename... Args>
  cl_int operator()(Args... arguments)
  {
    _args.reset();
    this->configure_arguments(arguments...);
    cl_int result = enqueue_kernel();
    _args.reset();

    return result;
  }

  template<typename... Args>
  void partial_argument_list(Args... arguments)
  {
    this->configure_arguments(arguments);
  }

  void discard_partial_arguments()
  {
    _args.reset();
  }

  cl_int enqueue_kernel() const
  {
    return this->_ctx->enqueue_ndrange_kernel(_kernel,
                                              _work_dim, _group_dim,
                                              _evt, cl::NullRange, _dependencies);
  }

private:
  template<typename T, typename... Args>
  void configure_arguments(const T&x, Args... arguments)
  {
    this->push_argument(x);
    this->configure_arguments(arguments...);
  }

  void configure_arguments()
  {}

  template<class T>
  void push_argument(const T& x)
  {
    _args.push(x);
  }

  template<class T>
  void push_argument(const local_memory<T>& x)
  {
    _args.push(nullptr, x.get_size());
  }

  template<class T>
  void push_argument(const raw_memory<T>& x)
  {
    _args.push(x.get_data(), x.get_size());
  }

  qcl::device_context_ptr _ctx;
  qcl::kernel_ptr _kernel;

  kernel_argument_list _args;
  cl::NDRange _work_dim;
  cl::NDRange _group_dim;

  cl::Event* _evt;
  std::vector<cl::Event>* _dependencies;
};


/// The global context represents a collection of devices (typically all
/// available devices for compute operations). It stores
/// one \c device_context object for each device.
class global_context
{
public:
  /// Create object
  /// \param contexts A vector containing \c device_context pointers that
  /// represent the devices that shall be managed by the \c global_context
  /// object.
  explicit global_context(const std::vector<device_context_ptr>& contexts)
  : _contexts(contexts), _active_device(0)
  {
  }
  
  /// Create \c global_context object based on only one device.
  /// \param context The \c device_context
  explicit global_context(const device_context_ptr& context)
  : _contexts(1, context), _active_device(0)
  {
  }
  
  // Object is uncopyable.
  global_context(const global_context& other) = delete;
  global_context& operator=(const global_context& other) = delete;
  
  /// \return The number of devices in the global context
  std::size_t get_num_devices() const
  {
    return _contexts.size();
  }
  
  /// Sets the default device returned by \c device()
  /// \param device The index of the device
  void set_active_device(std::size_t device)
  {
    assert(device < _contexts.size());
    _active_device = device;
  }
  
  /// Compiles a source file and registers kernels for all devices in
  /// the global context.
  /// \param cl_source_file The path the OpenCL source file
  /// \param kernel_names The names of the kernels within the source file
  void global_register_source_file(const std::string& cl_source_file, 
                               const std::vector<std::string>& kernel_names)
  {
    for(std::size_t i = 0; i < _contexts.size(); ++i)
      _contexts[i]->register_source_file(cl_source_file, kernel_names);
  }
  
  /// Compiles OpenCL source code and registers kernels for all devices in
  /// the global context.
  /// \param cl_source The path the OpenCL source code
  /// \param kernel_names The names of the kernels within the source code
  void global_register_source_code(const std::string& cl_source,
                    const std::vector<std::string>& kernel_names)
  {
    for(std::size_t i = 0; i < _contexts.size(); ++i)
      _contexts[i]->register_source_code(cl_source, kernel_names);
  }

  /// Compiles a QCL source module and registers kernels for all devices in
  /// the global context.
  /// \tparam Source_module_type The QCL source module
  /// \param kernel_names The names of the kernels within the source file
  template<class Source_module_type>
  void global_register_source_module(const std::vector<std::string>& kernel_names)
  {
    for(std::size_t i = 0; i < _contexts.size(); ++i)
      _contexts[i]->register_source_module<Source_module_type>(kernel_names);
  }
  
  /// \return The currently active device
  const device_context_ptr&
  device() const
  {
    return _contexts[_active_device];
  }
  
  /// \return The device with the given index
  /// \param dev The device index
  const device_context_ptr&
  device(std::size_t dev) const
  {
    assert(dev < _contexts.size());
    return _contexts[dev];
  }
  
private:
  std::vector<device_context_ptr> _contexts;
  
  std::size_t _active_device;
};

using global_context_ptr = std::shared_ptr<global_context>;
using const_global_context_ptr = std::shared_ptr<const global_context>;

/// Represents the OpenCL environment and as such provides methods
/// to query available platforms, devices and provides mechanisms
/// to construct \c global_context objects.
class environment
{
public:
  /// Initializes the object
  environment()
  {
    check_cl_error(cl::Platform::get(&_platforms), "Could not obtain Platform list!");
  }
  
  /// \return The available OpenCL platforms on this machine
  const std::vector<cl::Platform>& get_platforms() const
  {
    return _platforms;
  }
  
  /// \return The number of available OpenCL platforms
  std::size_t get_num_platforms() const
  {
    return _platforms.size();
  }
  
  /// \return The OpenCL platform with the given index
  /// \param idx The index of the platform
  const cl::Platform& get_platform(std::size_t idx) const
  {
    assert(idx < _platforms.size());
    return _platforms[idx];
  }
  
  /// Selects an OpenCL platform based on a list of preferences.
  /// \return The selected platform.
  /// \param preference_keywords Contains keywords that will be matched with
  /// the platform names and vendors. Elements of \c preference_keywords with
  /// a lower index will be assigned a higher priority. If no platform matches
  /// any keyword, the first available platform will be returned.
  const cl::Platform&
  get_platform_by_preference(const std::vector<std::string>& preference_keywords) const
  {
    if(_platforms.size() == 0)
      throw std::runtime_error("No available OpenCL platforms!");
    
    for(const std::string& keyword : preference_keywords)
    {
      for (const cl::Platform& platform : _platforms)
      {
        std::string platform_vendor = get_platform_vendor(platform);
        std::string platform_name = get_platform_name(platform);

        std::vector<cl::Device> devices;
        this->get_devices(platform, devices);

        // No need to consider empty platforms
        if(!devices.empty())
        {
          bool keyword_found_in_name = platform_name.find(keyword) != std::string::npos;
          bool keyword_found_in_vendor = platform_vendor.find(keyword) != std::string::npos;
          if (keyword_found_in_name || keyword_found_in_vendor)
            return platform;
        }
      }
    }

    return _platforms[0];
  }

  /// \return The vendor of a given platform
  /// \param platform The platform
  static
  std::string get_platform_vendor(const cl::Platform& platform)
  {
    std::string platform_vendor;
    platform.getInfo(static_cast<cl_platform_info>(CL_PLATFORM_VENDOR), &platform_vendor);
    return platform_vendor;
  }
  
  /// \return The name of a given platform
  /// \param platform The platform
  static
  std::string get_platform_name(const cl::Platform& platform)
  {
    std::string platform_name;
    platform.getInfo(static_cast<cl_platform_info>(CL_PLATFORM_NAME), &platform_name);
    return platform_name;
  }
  
  /// Creates a device context for a given platform and device
  /// \return The device context
  /// \param platform The platform
  /// \param device The device
  device_context_ptr create_device_context(const cl::Platform& platform, 
                                           const cl::Device& device) const
  {
    return device_context_ptr(new device_context(platform, device));
  }
  
  /// Creates a global context containing all devices from a given platform
  /// which are of a given device type.
  /// \return The global context
  /// \param platform The platform
  /// \param type The type of the device that shall be included in the context.
  /// If omitted, all available devices of the platform will be used.
  global_context_ptr create_global_context(const cl::Platform& platform, 
                                          cl_device_type type = CL_DEVICE_TYPE_ALL) const
  {
    std::vector<cl::Device> devices;
    get_devices(platform, devices, type);

    std::vector<device_context_ptr> contexts;
    
    for(std::size_t j = 0; j  < devices.size(); ++j)
    {
      device_context_ptr new_context(new device_context(platform, devices[j]));

      contexts.push_back(new_context);
    }
    
    global_context_ptr global_ctx(new global_context(contexts));
    return global_ctx;
  }
  
  /// Creates a global context containing all devices from all platforms
  /// which are of a given device type.
  /// \return The global context
  /// \param type The type of the device that shall be included in the context.
  /// If omitted, all available devices will be used.
  global_context_ptr create_global_context(cl_device_type type = CL_DEVICE_TYPE_ALL) const
  {
    std::vector<device_context_ptr> contexts;
    
    for(std::size_t i = 0; i < _platforms.size(); ++i)
    {
      std::vector<cl::Device> devices;
      get_devices(_platforms[i], devices, type);
      
      for(std::size_t j = 0; j  < devices.size(); ++j)
      {
        device_context_ptr new_context(new device_context(_platforms[i], devices[j]));
      
        contexts.push_back(new_context);
      }
    }
    
    global_context_ptr global_ctx(new global_context(contexts));
    
    return global_ctx;
  }
  
  /// Creates a global context containing all available GPUs of the system.
  /// \return The global context.
  global_context_ptr create_global_gpu_context() const
  {
    return create_global_context(CL_DEVICE_TYPE_GPU);
  }
  
  /// Creates a global context containing all available CPUs of the system.
  /// \return The global context.
  global_context_ptr create_global_cpu_context() const
  {
    return create_global_context(CL_DEVICE_TYPE_CPU);
  }
  
#ifndef WITHOUT_GL_INTEROP
  /// Creates a global context containing all devices which
  /// can share OpenCL objects with OpenGL.
  /// \return The global context.
  global_context_ptr create_global_gl_shared_context() const
  {
    std::vector<device_context_ptr> found_contexts;
    get_gl_sharable_contexts(found_contexts);
    
    global_context_ptr global_ctx(new global_context(found_contexts));
    return global_ctx;
  }
  
  static
  global_context_ptr create_global_gl_shared_context(const cl::Platform& platform)
  {
    std::vector<device_context_ptr> found_contexts;
    get_gl_sharable_contexts(platform, found_contexts);
    
    global_context_ptr global_ctx(new global_context(found_contexts));
    return global_ctx;
  }
  
  static
  void get_gl_sharable_contexts(const cl::Platform& platform,
                                std::vector<device_context_ptr>& contexts)
  {
    contexts.clear();
    
    cl_context_properties props [] =
    {
#if defined WIN32
      CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentContext()),
      CL_WGL_HDC_KHR,    reinterpret_cast(cl_context_properties>(wglGetCurrentDC()),
#elif defined __APPLE__
      CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, 
              reinterpret_cast<cl_context_properties>(CGLGetShareGroup(CGLGetCurrentContext())),
#else
      CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(glXGetCurrentContext()),
      CL_GLX_DISPLAY_KHR,reinterpret_cast<cl_context_properties>(glXGetCurrentDisplay()),
#endif
      CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()),
      0,
    };

    std::vector<cl::Device> devices;
    get_devices(platform, devices);

    for(const cl::Device& device : devices)
    {
#ifdef __APPLE__
      std::string required_extension = "cl_APPLE_gl_sharing";
#else
      std::string required_extension = "cl_khr_gl_sharing";
#endif
      if(device_context::is_extension_supported(device, required_extension))
      {
        try
        {
          cl_int err;
          cl::Context ctx = cl::Context(
            device, 
            props,
            NULL,
            NULL,
            &err);

          if(err == CL_SUCCESS)
          {
            device_context_ptr new_context = 
              std::make_shared<device_context>(ctx, device);

            contexts.push_back(new_context);
          }
        }
        catch(...)
        {}
      }
    }
  }
  
  void get_gl_sharable_contexts(std::vector<device_context_ptr>& contexts) const
  {
    contexts.clear();
    
    for(const cl::Platform& platform : _platforms)
    {
      std::vector<device_context_ptr> sharable_contexts_for_platform;
      get_gl_sharable_contexts(platform, sharable_contexts_for_platform);
      
      contexts.insert(contexts.end(), 
                      sharable_contexts_for_platform.begin(),
                      sharable_contexts_for_platform.end());
    }
  }
#endif
  
  inline static
  void get_devices(const cl::Platform& platform, 
                  std::vector<cl::Device>& result,
                  cl_device_type type = CL_DEVICE_TYPE_ALL)
  {
    result.clear();
    cl_int err = platform.getDevices(type, &result);
    // It should not be treated as error when there are no devices
    // on this platform
    if(err != CL_DEVICE_NOT_FOUND)
      check_cl_error(err, "Could not obtain device list!");
  }
  
  inline
  void get_devices(std::size_t platform_index, 
                  std::vector<cl::Device>& result,
                  cl_device_type type = CL_DEVICE_TYPE_ALL) const
  {
    assert(platform_index < _platforms.size());
    get_devices(_platforms[platform_index], result, type);
  }
  
  inline static
  void get_all_devices(const cl::Platform& platform,
                       std::vector<cl::Device>& result)
  { get_devices(platform, result); }
  
  inline
  void get_all_devices(std::size_t platform_index,
                       std::vector<cl::Device>& result) const
  { get_all_devices(_platforms[platform_index], result); }
  
  inline static
  void get_cpu_devices(const cl::Platform& platform,
                       std::vector<cl::Device>& result)
  { get_devices(platform, result, CL_DEVICE_TYPE_CPU); }
  
  inline
  void get_cpu_devices(std::size_t platform_index,
                       std::vector<cl::Device>& result) const
  { get_devices(platform_index, result, CL_DEVICE_TYPE_CPU); }
  
  inline static
  void get_gpu_devices(const cl::Platform& platform,
                       std::vector<cl::Device>& result)
  { get_devices(platform, result, CL_DEVICE_TYPE_GPU); }
  
  inline
  void get_gpu_devices(std::size_t platform_index,
                       std::vector<cl::Device>& result) const
  { get_devices(platform_index, result, CL_DEVICE_TYPE_GPU); }
  
private:
  std::vector<cl::Platform> _platforms;
};

}

#endif
