# QuickCL - a simple, header-only OpenCL wrapper to make OpenCL development quicker and more flexible

## Features
QuickCL can greatly simplify development of OpenCL acclerated software. Among its features are:
  * Automatical setup of all necessary OpenCL objects (command queues etc) for several devices. QuickCL provides convenient methods to select the devices you wish to compute on (e.g. by platform or by device type) and will automatically create all necessary objects to get you started.
  * Compiling source code or files with one simple function call. Compiled programs are cached and kernel can be easily retrieved by their name.
  * Convenient wrapper functions to copy buffers or create buffers (in particular, functions that work in terms of the number of elements and not in terms of the number of bytes - no more bugs due to forgottes multiplications by sizeof(T))
  * Still retains all of the flexibility of the OpenCL API by giving you access to the underlying objects, should you need them.
  * Supports the concept of QCL code modules. A module is a container for OpenCL code with powerful features:
    * Modules allow for a simple definition of OpenCL code within the same source file as the rest of you C++ application. 
    * Modules can be templated, and template arguments can easily forwarded into your OpenCL code. This makes it very convenient e.g. to implement OpenCL code for different dimensions or datatypes.
    * Kernels in modules can be called almost like normal functions!

## Building
QuickCL is header-only library and can simply by copied into the directory of your source code. Include qcl.hpp and qcl_module.hpp and you are good to go.

## Requirements
  * A C++11 compliant compiler
  * OpenCL with an (at least) OpenCL 1.2 compliant runtime API
  * The CL/cl2.hpp OpenCL C++ bindings. If your OpenCL distribution does come without it, you can find them here: https://github.com/KhronosGroup/OpenCL-CLHPP
  
## Getting started
Please see the file `example.cpp' for a quick demonstration on how to use QuickCL.

