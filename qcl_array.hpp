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

#ifndef QCL_ARRAY_HPP
#define QCL_ARRAY_HPP

#include "qcl.hpp"

#include <iterator>
#include <cassert>

namespace qcl {
namespace detail {

template<class T, class Array_type>
class array_iterator
{
public:
  array_iterator()
    : _obj{nullptr}, _pos{0}
  {}

  using array_type    = Array_type;
  using array_pointer = Array_type*;
  using array_reference = Array_type&;

  using value_type = T;
  using pointer    = T*;
  using reference  = T&;
  using difference_type = ptrdiff_t;

  array_iterator(const array_iterator&) = default;
  array_iterator& operator=(const array_iterator&) = default;

  explicit array_iterator(array_pointer object, std::size_t position)
    : _obj{object}, _pos{position}
  {}

  array_iterator& operator+=(difference_type n) noexcept
  { this->_pos += n; return *this; }

  array_iterator& operator-=(difference_type n) noexcept
  { this->_pos -= n; return *this; }


  array_iterator& operator++() noexcept
  { return *this += 1; }

  array_iterator operator++(int) noexcept
  {
    array_iterator copy{*this};
    (*this) += 1;
    return copy;
  }

  array_iterator& operator--() noexcept
  { return *this -= 1; }

  array_iterator operator--(int) noexcept
  {
    array_iterator copy{*this};
    (*this) -= 1;
    return copy;
  }


  array_pointer array() const noexcept
  {
    return _obj;
  }

  friend void swap(array_iterator& a,
                   array_iterator& b) noexcept
  {
    using std::swap;
    std::swap(a._obj, b._obj);
    std::swap(a._pos, b._pos);
  }

  std::size_t get_position() const noexcept
  {
    return _pos;
  }
private:
  array_pointer _obj;
  std::size_t _pos;
};


template<class T, class Array_type>
class array_iterator<const T, const Array_type>
{
public:
  array_iterator()
    : _obj{nullptr}, _pos{0}
  {}

  using array_type    = const Array_type;
  using array_pointer = const Array_type*;
  using array_reference = const Array_type&;

  using value_type = const T;
  using pointer    = const T*;
  using reference  = const T&;
  using difference_type = ptrdiff_t;

  /// Additional constructor to allow construction
  /// from the non-const version
  array_iterator(const array_iterator<T,Array_type>& non_const_other)
    : _obj{non_const_other.array()},
      _pos{non_const_other.get_position()}
  {}

  array_iterator(const array_iterator&) = default;
  array_iterator& operator=(const array_iterator&) = default;

  /// Additional assignment operator to allow assignment from
  /// the non-const version
  array_iterator& operator=(const array_iterator<T, Array_type>& non_const_other)
  {
    this->_obj = non_const_other.array();
    this->_pos = non_const_other.get_position();
  }

  explicit array_iterator(array_pointer object, std::size_t position)
    : _obj{object}, _pos{position}
  {}

  array_iterator& operator+=(difference_type n) noexcept
  { this->_pos += n; return *this; }

  array_iterator& operator-=(difference_type n) noexcept
  { this->_pos -= n; return *this; }


  array_iterator& operator++() noexcept
  { return *this += 1; }

  array_iterator operator++(int) noexcept
  {
    array_iterator copy{*this};
    (*this) += 1;
    return copy;
  }

  array_iterator& operator--() noexcept
  { return *this -= 1; }

  array_iterator operator--(int) noexcept
  {
    array_iterator copy{*this};
    (*this) -= 1;
    return copy;
  }

  array_pointer array() const noexcept
  {
    return _obj;
  }

  friend void swap(array_iterator& a,
                   array_iterator& b) noexcept
  {
    using std::swap;
    std::swap(a._obj, b._obj);
    std::swap(a._pos, b._pos);
  }

  std::size_t get_position() const noexcept
  {
    return _pos;
  }
private:
  array_pointer _obj;
  std::size_t _pos;
};

template<class T, class Array_type>
typename array_iterator<T,Array_type>::difference_type
operator-(const array_iterator<T,Array_type>& a,
          const array_iterator<T,Array_type>& b) noexcept
{ return a.get_position() - b.get_position(); }

template<class T, class Array_type>
bool
operator==(const array_iterator<T,Array_type>& a,
           const array_iterator<T,Array_type>& b) noexcept
{ return a.array() == b.array() && a.get_position() == b.get_position(); }

template<class T, class Array_type>
bool operator!=(const array_iterator<T,Array_type>& a,
                const array_iterator<T,Array_type>& b) noexcept
{ return !(a == b); }

template<class T, class Array_type>
bool operator<(const array_iterator<T,Array_type>& a,
               const array_iterator<T,Array_type>& b) noexcept
{ return a.get_position() < b.get_position(); }

template<class T, class Array_type>
bool operator<=(const array_iterator<T,Array_type>& a,
                const array_iterator<T,Array_type>& b) noexcept
{ return a.get_position() <= b.get_position(); }

template<class T, class Array_type>
bool operator>(const array_iterator<T, Array_type>& a,
               const array_iterator<T, Array_type>& b) noexcept
{ return a.get_position() > b.get_position(); }

template<class T, class Array_type>
bool operator>=(const array_iterator<T,Array_type>& a,
                const array_iterator<T,Array_type>& b) noexcept
{ return a.get_position() >= b.get_position(); }

template<class T, class Array_type>
array_iterator<T,Array_type>
operator+(const array_iterator<T,Array_type>& a,
          typename array_iterator<T,Array_type>::difference_type n) noexcept
{
  array_iterator<T,Array_type> it = a;
  it += n;

  return it;
}

template<class T, class Array_type>
array_iterator<T,Array_type>
operator+(typename array_iterator<T,Array_type>::difference_type n,
          const array_iterator<T,Array_type>& a) noexcept
{ return  a + n; }

template<class T, class Array_type>
array_iterator<T,Array_type>
operator-(const array_iterator<T,Array_type>& a,
          typename array_iterator<T,Array_type>::difference_type n) noexcept
{
  array_iterator<T,Array_type> it = a;
  it -= n;

  return it;
}

template<class T, class Array_type>
array_iterator<T,Array_type>
operator-(typename array_iterator<T,Array_type>::difference_type n,
          const array_iterator<T,Array_type>& a) noexcept
{ return  a - n; }

}

template<class T>
class device_array
{
public:
  using remote_iterator =
    detail::array_iterator<T, device_array<T>>;
  using const_remote_iterator =
    detail::array_iterator<const T, const device_array<T>>;

  device_array()
    : _num_elements{0}
  {}

  explicit device_array(const qcl::device_context_ptr& ctx,
                        const cl::Buffer& buff,
                        std::size_t num_elements)
    : _ctx{ctx}, _buff{buff}, _num_elements{num_elements}
  {}

  explicit device_array(const qcl::device_context_ptr& ctx,
                        const std::vector<T>& initial_data)
    : _ctx{ctx}, _num_elements{initial_data.size()}
  {
    assert(initial_data.size() > 0);

    _ctx->create_buffer<T>(_buff, initial_data.size());
    this->write(initial_data);
  }


  explicit device_array(const qcl::device_context_ptr& ctx,
                        std::size_t num_elements)
    : _ctx{ctx}, _num_elements{num_elements}
  {
    _ctx->create_buffer<T>(_buff, num_elements);
  }

  std::size_t size() const noexcept
  {
    return _num_elements;
  }

  const cl::Buffer& get_buffer() const noexcept
  {
    return _buff;
  }

  void read(std::vector<T>& out, command_queue_id queue = 0) const
  {
    out.resize(this->_num_elements);
    this->read(out.data(), begin(), end(), queue);
  }

  void read_async(std::vector<T>& out,
                  cl::Event* evt = nullptr,
                  std::vector<cl::Event>* dependencies = nullptr,
                  command_queue_id queue = 0) const
  {
    out.resize(this->_num_elements);

    this->read_async(out.data(),
                     begin(), end(),
                     evt,
                     dependencies,
                     queue);
  }

  void read(T* out,
            const_remote_iterator begin,
            const_remote_iterator end,
            command_queue_id queue = 0) const
  {
    _ctx->memcpy_d2h(out,
                     _buff,
                     begin.get_position(),
                     end.get_position(),
                     queue);
  }

  void read_async(T* out,
            const_remote_iterator begin,
            const_remote_iterator end,
            cl::Event* evt = nullptr,
            std::vector<cl::Event>* dependencies = nullptr,
            command_queue_id queue = 0) const
  {
    _ctx->memcpy_d2h_async(out.data(),
                           _buff,
                           begin.get_position(),
                           end.get_position(),
                           evt,
                           dependencies,
                           queue);
  }

  remote_iterator begin() noexcept
  { return remote_iterator{this,0}; }

  remote_iterator end() noexcept
  { return remote_iterator{this,_num_elements}; }

  const_remote_iterator begin() const noexcept
  { return const_remote_iterator{this, 0}; }

  const_remote_iterator end() const noexcept
  { return const_remote_iterator{this, _num_elements}; }

  void write(const T* data,
             remote_iterator out_begin, remote_iterator out_end,
             command_queue_id queue = 0)
  {
    assert(out_begin.array() == this && out_end.array() == this);
    _ctx->memcpy_h2d(_buff, data,
                     out_begin.get_position(),
                     out_end.get_position(),
                     queue);
  }

  void write_async(const T* data,
                   remote_iterator out_begin,
                   remote_iterator out_end,
                   cl::Event* evt = nullptr,
                   std::vector<cl::Event>* dependencies = nullptr,
                   command_queue_id queue = 0)
  {
    assert(out_begin.array() == this && out_end.array() == this);
    _ctx->memcpy_h2d_async(_buff, data,
                           out_begin.get_position(),
                           out_end.get_position(),
                           evt,
                           dependencies,
                           queue);
  }

  void write(const std::vector<T>& data,
             command_queue_id queue = 0)
  {
    assert(_num_elements >= data.size());
    write(data.data(),
          begin(), end(),
          queue);
  }

  void write_async(const std::vector<T>& data,
                   cl::Event* evt = nullptr,
                   std::vector<cl::Event>* dependencies = nullptr,
                   command_queue_id queue = 0)
  {
    assert(_num_elements >= data.size());
    write_async(data.data(),
                begin(), end(),
                evt, dependencies, queue);
  }
private:

  cl::Buffer _buff;

  qcl::device_context_ptr _ctx;
  std::size_t _num_elements;
};

namespace  detail {

/// This overload allows passing device_array objects
/// directly to qcl kernel calls via \c qcl::kernel_argument_list
/// and \c qcl::kernel_call
template<class T>
cl_int set_kernel_arg(std::size_t pos,
                      const kernel_ptr& kernel,
                      const device_array<T>& data)
{
  return kernel->setArg(pos, data.get_buffer());
}

}

}

#endif

