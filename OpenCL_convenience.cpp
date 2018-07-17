#include <stdio.h>  
#include <stdlib.h>  
#include <iostream>  
#include <CL/cl.h> 

#include "opencl_convenience.h"

#ifndef OPENCL_CONVENIENCE
#define OPENCL_CONVENIENCE

static inline int getNDRangeDim(int x, int y)
{
	return (x + y - 1) / y;
}

#endif /* OPENCL_CONVENIENCE */
