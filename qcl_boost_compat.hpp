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

#ifndef QCL_BOOST_COMPUTE_COMPAT_HPP
#define QCL_BOOST_COMPUTE_COMPAT_HPP

#define BOOST_COMPUTE_MEM_LEAK_WORKAROUND

#ifdef BOOST_COMPUTE_MEM_LEAK_WORKAROUND
#include "workaround/boost_compute_mem_leak.hpp"
#endif

#include <boost/compute.hpp>

#include "qcl.hpp"

namespace qcl {

template<class T>
static
boost::compute::buffer_iterator<T> create_buffer_iterator(const cl::Buffer& buffer,
                                                          std::size_t position)
{
  return boost::compute::make_buffer_iterator<T>(boost::compute::buffer{buffer.get()},
                                                 position);
}

boost::compute::context
static
create_boost_compute_context(const device_context_ptr& ctx)
{
  return boost::compute::context{ctx->get_context().get()};
}

template<class T>
struct to_boost_vector_type
{};

#define DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_type, boost_type) \
  template<> \
  struct to_boost_vector_type<cl_type> \
  { \
    using type = boost_type; \
  }

DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_float2, boost::compute::float2_);
DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_float4, boost::compute::float4_);
DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_float8, boost::compute::float8_);
DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_float16, boost::compute::float16_);

DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_double2, boost::compute::double2_);
DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_double4, boost::compute::double4_);
DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_double8, boost::compute::double8_);
DECLARE_BOOST_VECTOR_TYPE_TRANSLATOR(cl_double16, boost::compute::double16_);

}

#endif
