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


#ifndef QCL_MODULE_HPP
#define QCL_MODULE_HPP

#include <string>
#include <memory>

#include <boost/preprocessor/stringize.hpp>
#include <CL/cl2.hpp>

#include "qcl.hpp"

/// \file The macros in this file allow the use of QCL modules.
/// A module is a class that exports CL source code and uses
/// its class name as identifier to differentiate between kernels
/// from other modules.
/// Modules can include other modules, and allow the definition
/// of entrypoints which enable calling kernels with a single
/// appropriately named function call. These easy-to-use kernel
/// calls also feature automatic, on-demand compilation of the source.
/// Template arguments can be used as well.
///
/// Modules come in two variants:
/// * An existing class/struct can be turned into a module. This requires
/// at least a call to \c QCL_MAKE_MODULE(class name) and \c QCL_MAKE_SOURCE(source)
/// * Standalone modules. These require a call to \c QCL_STANDALONE_MODULE(module name)
/// and must be terminated by a call to \c QCL_STANDALONE_SOURCE(source code).
///
/// A standalone module may look like this:
/// \code
/// template<class T,int Scale>
/// QCL_STANDALONE_MODULE(my_module) // begin module
/// QCL_ENTRYPOINT(my_kernel_a) // make direct kernel calls available
/// QCL_ENTRYPOINT(my_kernel_b)
/// QCL_STANDALONE_SOURCE
/// (
///   QCL_INCLUDE_MODULE(some_other_module)
///   QCL_IMPORT_TYPE(T) // Make use of template parameter T
///   QCL_IMPORT_CONSTANT(Scale) // and Scale
///   QCL_RAW(
///    __kernel void my_kernel_a(__global T* a, __global T* b)
///    {
///      //Some more CL code here
///    }
///    __kernel void my_kernel_b(__global int* a, __global int* b)
///    {
///      int gid = get_global_id(0);
///      a[gid] = Scale * b[gid];
///    }
///   )
/// )
/// \endcode
///
/// The kernels can now be executed by calling
/// \code
/// qcl::device_context_ptr ctx = ...
/// cl::Buffer a = ...
/// cl::Buffer b = ...
/// my_module<T,Scale>::my_kernel_a(ctx, /*work dim*/), /*group dim*/)(a, b);
/// \endcode
/// and similarly for \c my_kernel_b. Note that these calls are always asynchronous (
/// you can pass a cl::Event pointer if you want to wait for them to complete).
/// You do \bold not need to explicitly compile the source code before executing these
/// calls, the source will be compiled the first time they are executed.
///
/// You can also extend an existing class into a source module (this
/// may be more convenient, if you have part of the algorithm already
/// encapsuled in this class). This can be done like this:
/// \code
/// class my_class
/// {
/// public:
///   my_class();
///
///   void existing_function();
///
///   QCL_MAKE_MODULE(my_class)
///   // define entrypoints
///   QCL_ENTRYPOINT(my_kernel_a)
///   QCL_MAKE_SOURCE(
///     // Optionally, include other modules
///     // or import constants as before
///     QCL_RAW(
///     __kernel void my_kernel_a(...){...}
///     )
///   )
/// private:
///   // attributes
/// };
/// \endcode
/// In this case, it is important:
///   - to use \c QCL_MAKE_MODULE(), \c QCL_MAKE_SOURCE() instead of
///     \c QCL_STANDALONE_MODULE() and \c QCL_STANDALONE_SOURCE()
///   - The macros must be called in the public section of your class.
///
/// For more details, see the documentation of the respective macros.

namespace qcl {
namespace detail {

template<class T>
struct cl_type_translator
{
};

#define DECLARE_TYPE_TRANSLATOR(T, cl_value) \
  template<> struct cl_type_translator<T>    \
  { static constexpr const char* value = BOOST_PP_STRINGIZE(cl_value); }

DECLARE_TYPE_TRANSLATOR(char,     char);
DECLARE_TYPE_TRANSLATOR(short,    short);
DECLARE_TYPE_TRANSLATOR(int,      int);
DECLARE_TYPE_TRANSLATOR(long,     int);
DECLARE_TYPE_TRANSLATOR(long long,long);

DECLARE_TYPE_TRANSLATOR(unsigned char,     uchar);
DECLARE_TYPE_TRANSLATOR(unsigned short,    ushort);
DECLARE_TYPE_TRANSLATOR(unsigned,          uint);
DECLARE_TYPE_TRANSLATOR(unsigned long,     uint);
DECLARE_TYPE_TRANSLATOR(unsigned long long,ulong);

DECLARE_TYPE_TRANSLATOR(float, float);
DECLARE_TYPE_TRANSLATOR(double, double);

DECLARE_TYPE_TRANSLATOR(cl_float2,  float2);
DECLARE_TYPE_TRANSLATOR(cl_float4,  float4);
DECLARE_TYPE_TRANSLATOR(cl_float8,  float8);
DECLARE_TYPE_TRANSLATOR(cl_float16, float16);

DECLARE_TYPE_TRANSLATOR(cl_double2,  double2);
DECLARE_TYPE_TRANSLATOR(cl_double4,  double4);
DECLARE_TYPE_TRANSLATOR(cl_double8,  double8);
DECLARE_TYPE_TRANSLATOR(cl_double16, double16);

DECLARE_TYPE_TRANSLATOR(cl_uchar2,  uchar2);
DECLARE_TYPE_TRANSLATOR(cl_uchar4,  uchar4);
DECLARE_TYPE_TRANSLATOR(cl_uchar8,  uchar8);
DECLARE_TYPE_TRANSLATOR(cl_uchar16, uchar16);

DECLARE_TYPE_TRANSLATOR(cl_char2,  char2);
DECLARE_TYPE_TRANSLATOR(cl_char4,  char4);
DECLARE_TYPE_TRANSLATOR(cl_char8,  char8);
DECLARE_TYPE_TRANSLATOR(cl_char16, char16);

DECLARE_TYPE_TRANSLATOR(cl_short2,  short2);
DECLARE_TYPE_TRANSLATOR(cl_short4,  short4);
DECLARE_TYPE_TRANSLATOR(cl_short8,  short8);
DECLARE_TYPE_TRANSLATOR(cl_short16, short16);

DECLARE_TYPE_TRANSLATOR(cl_ushort2,  ushort2);
DECLARE_TYPE_TRANSLATOR(cl_ushort4,  ushort4);
DECLARE_TYPE_TRANSLATOR(cl_ushort8,  ushort8);
DECLARE_TYPE_TRANSLATOR(cl_ushort16, ushort16);

DECLARE_TYPE_TRANSLATOR(cl_int2,  int2);
DECLARE_TYPE_TRANSLATOR(cl_int4,  int4);
DECLARE_TYPE_TRANSLATOR(cl_int8,  int8);
DECLARE_TYPE_TRANSLATOR(cl_int16, int16);

DECLARE_TYPE_TRANSLATOR(cl_uint2,  uint2);
DECLARE_TYPE_TRANSLATOR(cl_uint4,  uint4);
DECLARE_TYPE_TRANSLATOR(cl_uint8,  uint8);
DECLARE_TYPE_TRANSLATOR(cl_uint16, uint16);

DECLARE_TYPE_TRANSLATOR(cl_long2,  long2);
DECLARE_TYPE_TRANSLATOR(cl_long4,  long4);
DECLARE_TYPE_TRANSLATOR(cl_long8,  long8);
DECLARE_TYPE_TRANSLATOR(cl_long16, long16);

DECLARE_TYPE_TRANSLATOR(cl_ulong2,  ulong2);
DECLARE_TYPE_TRANSLATOR(cl_ulong4,  ulong4);
DECLARE_TYPE_TRANSLATOR(cl_ulong8,  ulong8);
DECLARE_TYPE_TRANSLATOR(cl_ulong16, ulong16);


}
}

/// This macro, together with \c QCL_MAKE_SOURCE is required
/// to turn an existing class into a QCL module. This macro
/// must be called in the public section of the class that should
/// become a qcl module.
/// \param module_name The name of the class
#define QCL_MAKE_MODULE(module_name)                           \
    typedef module_name _qcl_this_type;                        \
    static std::string _qcl_get_module_name()                  \
    {return typeid(_qcl_this_type).name();}


/// This macro, together with \c QCL_MAKE_MODULE is required
/// to turn an existing class into a QCL module. This macro
/// must be called in the public section of the class that should
/// become a qcl module.
/// This call can have
/// nested calls to \c QCL_INCLUDE_MODULE, \c QCL_IMPORT_TYPE and
/// \c QCL_IMPORT_CONSTANT. These calls must not be the last calls
/// of the argument: The last part must be either a call to \c QCL_RAW()
/// or a string literal with the source code. Example:
/// \code
/// QCL_MAKE_MODULE(...)
/// // possible entrypoints go here
/// QCL_MAKE_SOURCE(
///   QCL_INCLUDE_MODULE(other module)
///   "CL code goes here"
/// )
/// \endcode
/// \param source_code A string containing the CL source code.
#define QCL_MAKE_SOURCE(source_code) \
  static std::string _qcl_source()              \
  {                                             \
    std::string code = source_code;             \
    std::string include_guard = "QCL_MODULE_"+_qcl_get_module_name()+"_CL"; \
    return "#ifndef "+include_guard             \
        +"\n#define "+include_guard             \
        +"\n"+code                              \
        +"\n#endif\n";                          \
  }


/// Start a standalone module. Must be followed by \c QCL_STANDALONE_SOURCE()
/// to end the module.
#define QCL_STANDALONE_MODULE(module_name)                     \
  struct module_name {                                         \
    QCL_MAKE_MODULE(module_name)

/// Define source code for a standalone module. Must be preceded
/// by \c QCL_STANDALONE_MODULE(). This call can have
/// nested calls to \c QCL_INCLUDE_MODULE, \c QCL_IMPORT_TYPE and
/// \c QCL_IMPORT_CONSTANT. These calls must not be the last calls
/// of the argument: The last part must be either a call to \c QCL_RAW()
/// or a string literal with the source code. Example:
/// \code
/// QCL_STANDALONE_MODULE(...)
/// // possible entrypoints go here
/// QCL_STANDALONE_SOURCE(
///   QCL_INCLUDE_MODULE(other module)
///   "CL code goes here"
/// )
/// \endcode
/// \param source_code CL source code
#define QCL_STANDALONE_SOURCE(source_code)           \
   QCL_MAKE_SOURCE(source_code) \
 };

/// Defines an entrypoint. For standalone modules, must be called
/// between \c QCL_STANDALONE_MODULE() and \c QCL_STANDALONE_SOURCE().
///
/// Defines a function that compiles the source (if not already
/// compiled) and returns a \c qcl::kernel_call object to execute
/// the kernel.
/// \param kernel_name The entrypoint's name. Must correspond to
/// a kernel in the CL source
#define QCL_ENTRYPOINT(kernel_name) \
  static \
  qcl::kernel_call kernel_name(const qcl::device_context_ptr& ctx,  \
                               const cl::NDRange& minimum_work_dim, \
                               const cl::NDRange& group_dim,        \
                               cl::Event* evt = nullptr,            \
                               std::vector<cl::Event>* dependencies = nullptr) \
  { \
    std::string kernel_name = BOOST_PP_STRINGIZE(kernel_name);                \
    ctx->register_source_code(_qcl_source(),                                  \
                             std::vector<std::string>{kernel_name},           \
                             _qcl_get_module_name(),                          \
                             _qcl_get_module_name());                         \
    return qcl::kernel_call(ctx,                                              \
                            ctx->get_kernel(_qcl_get_module_name()+"::"+kernel_name),\
                            minimum_work_dim,                                 \
                            group_dim,                                        \
                            evt,                                              \
                            dependencies);                                    \
  }


/// Makes a different module accessible from the current CL module.
/// This call must be nested inside a \c QCL_STANDALONE_SOURCE() or
/// \c QCL_MAKE_SOURCE() call (see their documentation for more information).
/// Note that currently, there is no dedection of cyclic dependencies,
/// so be careful when including modules.
/// \param module The name of the source module that shall be included
#define QCL_INCLUDE_MODULE(module) module::_qcl_source()+

/// Make a template type argument accessible from the CL source.
/// This call must be nested inside a \c QCL_STANDALONE_SOURCE() or
/// \c QCL_MAKE_SOURCE() call (see their documentation for more information).
/// \param template_parameter The template parameter
#define QCL_IMPORT_TYPE(template_parameter) \
  std::string{"\n#define " BOOST_PP_STRINGIZE(template_parameter) " "} \
  + std::string(qcl::detail::cl_type_translator<template_parameter>::value) \
  + std::string{"\n"} +

/// Make a constant (or a non-type template) available in CL source
/// code.
/// This call must be nested inside a \c QCL_STANDALONE_SOURCE() or
/// \c QCL_MAKE_SOURCE() call (see their documentation for more information).
/// \param constant_name The name of the constant.
#define QCL_IMPORT_CONSTANT(constant_name) \
  std::string("\n#define " BOOST_PP_STRINGIZE(constant_name) " (") \
  + std::to_string(constant_name)+std::string{")\n"}+

/// Stringizes the argument, and hence allows to write CL code
/// (seemingly) outside the quotation marks of a string literal.
/// This helps syntax highlighting/code assistance for many IDEs.
/// However, note that in C++ stringized data is always stripped
/// of newlines. Not only does this make it more difficult to
/// interpret error messages of the CL compiler, in particular
/// it also breaks preprocessor macros in CL code.
/// This can be circumvented by using several calls to QCL_RAW,
/// as each one will be terminated by a newline, or by using
/// C++11 raw string literals instead.
#define QCL_RAW(source) BOOST_PP_STRINGIZE(source) "\n"
/// Stringizes the argument as single line and adds a newline.
#define QCL_SINGLE_LINE(source) "\n" QCL_RAW(source)
/// Defines a preprocessor command in qcl code. This avoids having
/// the preprocessor confused due to CL preprocessor commands that
/// start with # as well but should be stringized. Example:
/// \c QCL_PREPROCESSOR(define, NUMBER 1234) is equivalent to
/// \c #define NUMBER 1234
#define QCL_PREPROCESSOR(command, content) \
  "\n#" BOOST_PP_STRINGIZE(command) " " BOOST_PP_STRINGIZE(content) "\n"





#endif
