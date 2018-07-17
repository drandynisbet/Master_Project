#include <stdio.h>  
#include <stdlib.h>  
#include <iostream>  
#include <CL/cl.h> 


#ifndef OPENCL_CONVENIENCE
#define OPENCL_CONVENIENCE

static inline int getNDRangeDim(int x, int y)
{
	return (x + y - 1) / y;
}

static inline void cudaSafeCall(cudaError_t err)
{
	if (cudaSuccess != err)
	{
		std::cout << "Error: " << cudaGetErrorString(err) << ": " << __FILE__ << ":" << __LINE__ << std::endl;
		exit(0);
	}
}

#endif /* OPENCL_CONVENIENCE */
