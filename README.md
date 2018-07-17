# Master_Project

Errors:
1. The question of Kernel function floatn

Problem：Everything is all right whenthe code of host used 'glm::vec3' to set vector and kernel used custom structure 'struct vec3{float x; float y; float z;}'，
but when using OpenCL's float3, it can not be recongized.

Reason：Checked the source code of cl_platform.h，have find typedef  cl_float4  cl_float3; float3 were aligned by float4 .

Solution：Make the variable 4D or change the source code.


2.Stack overflow occurs when clBuildProgram() runs

Problem: The clBuildProgram() of host stack overflow error.

Reason：The kernel of device does not support the recursive function.

Solution：Change the rule of recursive function or write a stack in global memory.


3.The kernel of OpenCL does not support malloc.


4. The error of clCreateFromGLBuffer

Problem：Make program betweeen OpenCL and OpenGL, when using clCreateFromBuffer, it will be failed and the error code is -34.

Reason：The error is the setting of context, I need to use OpenGL's environments。

Solution：Using OpenCL's OpenGL to set context。
