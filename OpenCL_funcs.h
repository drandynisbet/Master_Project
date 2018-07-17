

#include "OpenCL_funcs.h"
#include "OpenCL_convenience.h"
#include "OpenCL_operators.h"


const char *pyrDownGaussKernelSource = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                        \n"\
"__kernel void pyrDownGauss(__global const PtrStepSz<unsigned short> src,       \n"\
"								 __global PtrStepSz<unsigned short> dst,              \n"\
"								 __global float sigma_color)                          \n"\
"{                                                                    \n"\
"	int idx = get_global_id(0);                                       \n"\
"	int idy = get_global_id(1);                                       \n"\
"                                                                     \n"\
"	if (idx >= dst.cols || idy >= dst.rows)                           \n"\
"		return;                                                       \n"\
"                                                                     \n"\
"	const int D = 5;                                                  \n"\
"                                                                     \n"\
"	int center = src.ptr(2 * y)[2 * x];                               \n"\
"                                                                     \n"\
"	int x_mi = max(0, 2 * idx - D / 2) - 2 * idx;                     \n"\
"	int y_mi = max(0, 2 * idy - D / 2) - 2 * idy;                     \n"\
"                                                                     \n"\
"	int x_ma = min(src.cols, 2 * idx - D / 2 + D) - 2 * idx;          \n"\
"	int y_ma = min(src.rows, 2 * idy - D / 2 + D) - 2 * idy;          \n"\
"                                                                     \n"\
"	float sum = 0;                                                    \n"\
"	float wall = 0;                                                   \n"\
"                                                                     \n"\
"	float weights[] = { 0.375f, 0.25f, 0.0625f };                     \n"\
"                                                                     \n"\
"	for (int yi = y_mi; yi < y_ma; ++yi)                              \n"\
"		for (int xi = x_mi; xi < x_ma; ++xi)                          \n"\
"		{                                                             \n"\
"			int val = src.ptr(2 * y + yi)[2 * x + xi];                \n"\
"                                                                     \n"\
"			if (abs(val - center) < 3 * sigma_color)                  \n"\
"			{                                                         \n"\
"				sum += val * weights[abs(xi)] * weights[abs(yi)];     \n"\
"				wall += weights[abs(xi)] * weights[abs(yi)];          \n"\
"			}                                                         \n"\
"		}                                                             \n"\
"                                                                     \n"\
"	dst.ptr(idy)[idx] = static_cast<int>(sum / wall);                 \n"\
"}\n"\
"\n";

int pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst)
{

	//cl_uint status;
	//cl_platform_id platform;

	////paltform
	//status = clGetPlatformIDs(1, &platform, NULL);
	//cl_device_id device;
	////device
	//clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,1,&device,NULL);
	////context
	//cl_context context = clCreateContext(NULL,1,&device,NULL, NULL, NULL);
	////command queue
	//cl_command_queue commandQueue = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE, NULL);

	//if (commandQueue == NULL)
	//	perror("Failed to create commandQueue for device 0.");

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1,(const char **)& pyrDownGaussKernelSource, NULL, &err);
	
	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pyrDownGauss", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,bytes, h_c, 0, NULL, NULL);
};

const char *computeVmapKernelSource = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                        \n"\
"__kernel void computeVmapKernel(__global const PtrStepSz<unsigned short> depth,      \n"\
"						   __global PtrStep<float> vmap, \n"\
"						   __global float fx_inv, \n"\
"						   __global float fy_inv, \n"\
"						   __global float cx, \n"\
"						   __global float cy, \n"\
"                          __global float depthCutoff) \n"\
"{\n"\
"	int idu = get_global_id(0);\n"\
"	int idv = get_global_id(1);\n"\
"\n"\
"	if (idu < depth.cols && idv < depth.rows)\n"\
"{\n"\
"	float z = depth.ptr(idv)[idu] / 1000.f; // load and convert: mm -> meters\n"\
"\n"\
"	if (z != 0 && z < depthCutoff)\n"\
"	{\n"\
"		float vx = z * (idu - cx) * fx_inv;\n"\
"		float vy = z * (idv - cy) * fy_inv;\n"\
"		float vz = z;\n"\
"\n"\
"		vmap.ptr(idv)[idu] = vx;\n"\
"		vmap.ptr(idv + depth.rows)[idu] = vy;\n"\
"		vmap.ptr(idv + depth.rows * 2)[idu] = vz;\n"\
"	}\n"\
"	else\n"\
"	{\n"\
"		vmap.ptr(idv)[idu] = __int_as_float(0x7fffffff); \n"\
"	}\n"\
"}\n"\
"}\n"\ 
"\n";


int createVMap(const CameraModel& intr, const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float> & vmap, const float depthCutoff)
{
	cl_platform_id cpPlatform;        
	cl_device_id device_id;          
	cl_context context;              
	cl_command_queue queue;          
	cl_program program;              
	cl_kernel kernel;                


	vmap.create(depth.rows() * 3, depth.cols());


	/*dim3 grid(1, 1, 1);
	grid.x = getGridDim(depth.cols(), block.x);
	grid.y = getGridDim(depth.rows(), block.y);*/

	float fx = intr.fx, cx = intr.cx;
	float fy = intr.fy, cy = intr.cy;

	//computeVmapKernel << <grid, block >> >(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff);
	//cudaSafeCall(cudaGetLastError());


	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& computeVmapKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pyrDownGauss", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

	
}

const char *computeNmapKernelSource = "\n" \
"__kernel void computeNmapKernel(int rows, \n" \
"								int cols, \n" \
"								__global const PtrStep<float> vmap, \n" \
"								__global PtrStep<float> nmap)\n" \
"{\n" \
"	int u = get_global_id(0);\n" \
"	int v = get_global_id(1);\n" \
"\n" \
"	if (u >= cols || v >= rows)\n" \
"		return;\n" \
"\n" \
"	if (u == cols - 1 || v == rows - 1)\n" \
"	{\n" \
"		nmap.ptr(v)[u] = __int_as_float(0x7fffffff); \n" \
"		return;\n" \
"	}\n" \
"\n" \
"	float3 v00, v01, v10;\n" \
"	v00.x = vmap.ptr(v)[u];\n" \
"	v01.x = vmap.ptr(v)[u + 1];\n" \
"	v10.x = vmap.ptr(v + 1)[u];\n" \
"\n" \
"	if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x))\n" \
"	{\n" \
"		v00.y = vmap.ptr(v + rows)[u];\n" \
"		v01.y = vmap.ptr(v + rows)[u + 1];\n" \
"		v10.y = vmap.ptr(v + 1 + rows)[u];\n" \
"\n" \
"		v00.z = vmap.ptr(v + 2 * rows)[u];\n" \
"		v01.z = vmap.ptr(v + 2 * rows)[u + 1];\n" \
"		v10.z = vmap.ptr(v + 1 + 2 * rows)[u];\n" \
"\n" \
"		float3 r = normalized(cross(v01 - v00, v10 - v00));\n" \
"\n" \
"		nmap.ptr(v)[u] = r.x;\n" \
"		nmap.ptr(v + rows)[u] = r.y;\n" \
"		nmap.ptr(v + 2 * rows)[u] = r.z;\n" \
"	}\n" \
"	else\n" \
"		nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /\n" \
"}
"\n";
int createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
	/*nmap.create(vmap.rows(), vmap.cols());

	int rows = vmap.rows() / 3;
	int cols = vmap.cols();

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(cols, block.x);
	grid.y = getGridDim(rows, block.y);

	computeNmapKernel << <grid, block >> >(rows, cols, vmap, nmap);
	cudaSafeCall(cudaGetLastError());*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;               
	cl_command_queue queue;           
	cl_program program;               
	cl_kernel kernel;                 


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& computeNmapKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "computeNmap", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
}

const char *tranformMapsKernelSource = "\n" \
"__kernel void tranformMapsKernel(int rows, \n" \
"								  int cols, \n" \
"								  const PtrStep<float> vmap_src, \n" \
"                                 const PtrStep<float> nmap_src,\n" \
"								  const mat33 Rmat, const float3 tvec, \n" \
"								  PtrStepSz<float> vmap_dst, \n" \
"                                 PtrStep<float> nmap_dst)\n" \
"{\n" \
"	int x = get_global_id(0);\n" \
"	int y = get_global_id(1);\n" \
"\n" \
"	if (x < cols && y < rows)\n" \
"	{\n" \
"		//vertexes\n" \
"		float3 vsrc, vdst = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));\n" \
"		vsrc.x = vmap_src.ptr(y)[x];\n" \
"\n" \
"		if (!isnan(vsrc.x))\n" \
"		{\n" \
"			vsrc.y = vmap_src.ptr(y + rows)[x];\n" \
"			vsrc.z = vmap_src.ptr(y + 2 * rows)[x];\n" \
"\n" \
"			vdst = Rmat * vsrc + tvec;\n" \
"\n" \
"			vmap_dst.ptr(y + rows)[x] = vdst.y;\n" \
"			vmap_dst.ptr(y + 2 * rows)[x] = vdst.z;\n" \
"		}\n" \
"\n" \
"		vmap_dst.ptr(y)[x] = vdst.x;\n" \
"\n" \
"		//normals\n" \
"		float3 nsrc, ndst = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));\n" \
"		nsrc.x = nmap_src.ptr(y)[x];\n" \
"\n" \
"		if (!isnan(nsrc.x))\n" \
"		{\n" \
"			nsrc.y = nmap_src.ptr(y + rows)[x];\n" \
"			nsrc.z = nmap_src.ptr(y + 2 * rows)[x];\n" \
"\n" \
"			ndst = Rmat * nsrc;\n" \
"\n" \
"			nmap_dst.ptr(y + rows)[x] = ndst.y;\n" \
"			nmap_dst.ptr(y + 2 * rows)[x] = ndst.z;\n" \
"		}\n" \
"\n" \
"		nmap_dst.ptr(y)[x] = ndst.x;\n" \
"	}\n" \
"}\n" \
"\n";


int tranformMaps(const DeviceArray2D<float>& vmap_src,
	const DeviceArray2D<float>& nmap_src,
	const mat33& Rmat, const float3& tvec,
	DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
	/*int cols = vmap_src.cols();
	int rows = vmap_src.rows() / 3;

	vmap_dst.create(rows * 3, cols);
	nmap_dst.create(rows * 3, cols);

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(cols, block.x);
	grid.y = getGridDim(rows, block.y);

	tranformMapsKernel << <grid, block >> >(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
	cudaSafeCall(cudaGetLastError());

	nmap.create(vmap.rows(), vmap.cols());

	int rows = vmap.rows() / 3;
	int cols = vmap.cols();

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(cols, block.x);
	grid.y = getGridDim(rows, block.y);

	computeNmapKernel << <grid, block >> >(rows, cols, vmap, nmap);
	cudaSafeCall(cudaGetLastError());*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;               
	cl_command_queue queue;          
	cl_program program;               
	cl_kernel kernel;                 


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& tranformMapsKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "tranformMapsKernel", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
}

const char *copyMapsKernelSource = "\n" \
"__kernel void copyMapsKernel(int rows,    \n" \
"	int cols,    \n" \
"	const float * vmap_src,    \n" \
"	const float * nmap_src,    \n" \
"	PtrStepSz<float> vmap_dst,    \n" \
"	PtrStep<float> nmap_dst)    \n" \
"{    \n" \
"	int x = get_global_id(0);    \n" \
"	int y = get_global_id(1);    \n" \
"    \n" \
"	if (x < cols && y < rows)    \n" \
"	{    \n" \
"		//vertexes    \n" \
"		float3 vsrc, vdst = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));    \n" \
"    \n" \
"		vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];    \n" \
"		vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];    \n" \
"		vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];    \n" \
"    \n" \
"		if (!(vsrc.z == 0))    \n" \
"		{    \n" \
"			vdst = vsrc;    \n" \
"		}    \n" \
"    \n" \
"		vmap_dst.ptr(y)[x] = vdst.x;    \n" \
"		vmap_dst.ptr(y + rows)[x] = vdst.y;    \n" \
"		vmap_dst.ptr(y + 2 * rows)[x] = vdst.z;    \n" \
"    \n" \
"		//normals    \n" \
"		float3 nsrc, ndst = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));    \n" \
"    \n" \
"		nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];    \n" \
"		nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];    \n" \
"		nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];    \n" \
"    \n" \
"		if (!(vsrc.z == 0))    \n" \
"		{    \n" \
"			ndst = nsrc;    \n" \
"		}    \n" \
"    \n" \
"		nmap_dst.ptr(y)[x] = ndst.x;    \n" \
"		nmap_dst.ptr(y + rows)[x] = ndst.y;    \n" \
"		nmap_dst.ptr(y + 2 * rows)[x] = ndst.z;    \n" \
"	}    \n" \
"}    \n" \
"\n";

int copyMaps(const DeviceArray<float>& vmap_src,
	const DeviceArray<float>& nmap_src,
	DeviceArray2D<float>& vmap_dst,
	DeviceArray2D<float>& nmap_dst)
{
	/*int cols = vmap_dst.cols();
	int rows = vmap_dst.rows() / 3;

	vmap_dst.create(rows * 3, cols);
	nmap_dst.create(rows * 3, cols);

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(cols, block.x);
	grid.y = getGridDim(rows, block.y);

	copyMapsKernel << <grid, block >> >(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
	cudaSafeCall(cudaGetLastError());

	int cols = vmap_src.cols();
	int rows = vmap_src.rows() / 3;

	vmap_dst.create(rows * 3, cols);
	nmap_dst.create(rows * 3, cols);

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(cols, block.x);
	grid.y = getGridDim(rows, block.y);

	tranformMapsKernel << <grid, block >> >(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
	cudaSafeCall(cudaGetLastError());*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;              
	cl_command_queue queue;          
	cl_program program;               
	cl_kernel kernel;                 


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& copyMapsKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "copyMapsKernel", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, , CL_TRUE, 0, bytes, , 0, NULL, NULL);


}



template<bool normalize>
const char *resizeMapKernelSource = "\n" \
"__kernel void resizeMapKernel(int drows,        \n" \
"								int dcols,         \n" \
"								int srows,         \n" \
"								const PtrStep<float> input,         \n" \
"								PtrStep<float> output)        \n" \
"{        \n" \
"	int x = threadIdx.x + blockIdx.x * blockDim.x;        \n" \
"	int y = threadIdx.y + blockIdx.y * blockDim.y;        \n" \
"        \n" \
"	if (x >= dcols || y >= drows)        \n" \
"		return;        \n" \
"        \n" \
"	const float qnan = __int_as_float(0x7fffffff);        \n" \
"        \n" \
"	int xs = x * 2;        \n" \
"	int ys = y * 2;        \n" \
"        \n" \
"	float x00 = input.ptr(ys + 0)[xs + 0];        \n" \
"	float x01 = input.ptr(ys + 0)[xs + 1];        \n" \
"	float x10 = input.ptr(ys + 1)[xs + 0];        \n" \
"	float x11 = input.ptr(ys + 1)[xs + 1];        \n" \
"        \n" \
"	if (isnan(x00) || isnan(x01) || isnan(x10) || isnan(x11))        \n" \
"	{        \n" \
"		output.ptr(y)[x] = qnan;        \n" \
"		return;        \n" \
"	}        \n" \
"	else        \n" \
"	{        \n" \
"		float3 n;        \n" \
"        \n" \
"		n.x = (x00 + x01 + x10 + x11) / 4;        \n" \
"        \n" \
"		float y00 = input.ptr(ys + srows + 0)[xs + 0];        \n" \
"		float y01 = input.ptr(ys + srows + 0)[xs + 1];        \n" \
"		float y10 = input.ptr(ys + srows + 1)[xs + 0];        \n" \
"		float y11 = input.ptr(ys + srows + 1)[xs + 1];        \n" \
"        \n" \
"		n.y = (y00 + y01 + y10 + y11) / 4;        \n" \
"        \n" \
"		float z00 = input.ptr(ys + 2 * srows + 0)[xs + 0];        \n" \
"		float z01 = input.ptr(ys + 2 * srows + 0)[xs + 1];        \n" \
"		float z10 = input.ptr(ys + 2 * srows + 1)[xs + 0];        \n" \
"		float z11 = input.ptr(ys + 2 * srows + 1)[xs + 1];        \n" \
"        \n" \
"		n.z = (z00 + z01 + z10 + z11) / 4;        \n" \
"        \n" \
"		if (normalize)        \n" \
"			n = normalized(n);        \n" \
"        \n" \
"		output.ptr(y)[x] = n.x;        \n" \
"		output.ptr(y + drows)[x] = n.y;        \n" \
"		output.ptr(y + 2 * drows)[x] = n.z;        \n" \
"	}        \n" \
"}        \n" \
"\n";

template<bool normalize>
int resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
	/*int in_cols = input.cols();
	int in_rows = input.rows() / 3;

	int out_cols = in_cols / 2;
	int out_rows = in_rows / 2;

	output.create(out_rows * 3, out_cols);

	dim3 block(32, 8);
	dim3 grid(getGridDim(out_cols, block.x), getGridDim(out_rows, block.y));
	resizeMapKernel<normalize> << < grid, block >> >(out_rows, out_cols, in_rows, input, output);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;               
	cl_command_queue queue;           
	cl_program program;               
	cl_kernel kernel;                 


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& resizeMapKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "resizeMapKernel", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
}

int resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
	resizeMap<false>(input, output);
}

int resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
	resizeMap<true>(input, output);
}

const char *pyrDownKernelGaussFSource = "\n" \
"__kernel void pyrDownKernelGaussF(const PtrStepSz<float> src, PtrStepSz<float> dst, float * gaussKernel)   \n" \
"{   \n" \
"	int x = get_global_id(0);   \n" \
"	int y = get_global_id(1);   \n" \
"   \n" \
"	if (x >= dst.cols || y >= dst.rows)   \n" \
"		return;   \n" \
"   \n" \
"	const int D = 5;   \n" \
"   \n" \
"	float center = src.ptr(2 * y)[2 * x];   \n" \
"   \n" \
"	int tx = min(2 * x - D / 2 + D, src.cols - 1);   \n" \
"	int ty = min(2 * y - D / 2 + D, src.rows - 1);   \n" \
"	int cy = max(0, 2 * y - D / 2);   \n" \
"   \n" \
"	float sum = 0;   \n" \
"	int count = 0;   \n" \
"   \n" \
"	for (; cy < ty; ++cy)   \n" \
"	{   \n" \
"		for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx)   \n" \
"		{   \n" \
"			if (!isnan(src.ptr(cy)[cx]))   \n" \
"			{   \n" \
"				sum += src.ptr(cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];   \n" \
"				count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];   \n" \
"			}   \n" \
"		}   \n" \
"	}   \n" \
"	dst.ptr(y)[x] = (float)(sum / (float)count);   \n" \
"}   \n" \
"\n";

int pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float> & dst)
{
	//dst.create(src.rows() / 2, src.cols() / 2);

	//dim3 block(32, 8);
	//dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

	//const float gaussKernel[25] = { 1, 4, 6, 4, 1,
	//	4, 16, 24, 16, 4,
	//	6, 24, 36, 24, 6,
	//	4, 16, 24, 16, 4,
	//	1, 4, 6, 4, 1 };

	//float * gauss_cuda;

	//cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
	//cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

	//pyrDownKernelGaussF << <grid, block >> >(src, dst, gauss_cuda);
	//cudaSafeCall(cudaGetLastError());

	//cudaFree(gauss_cuda);

	cl_platform_id cpPlatform;       
	cl_device_id device_id;           
	cl_context context;               
	cl_command_queue queue;           
	cl_program program;               
	cl_kernel kernel;                 

	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& pyrDownKernelGaussFSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pyrDownKernelGaussF", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, , CL_TRUE, 0, bytes, , 0, NULL, NULL);
};

const char *pyrDownKernelIntensityGaussSource = "\n" \
"__kernel void pyrDownKernelIntensityGauss(const PtrStepSz<unsigned char> src,     \n" \
"											PtrStepSz<unsigned char> dst,     \n" \
"											float * gaussKernel)     \n" \
"{     \n" \
"	int x = get_global_id(0);     \n" \
"	int y = get_global_id(1);     \n" \
"    \n" \
"	if (x >= dst.cols || y >= dst.rows)     \n" \
"		return;     \n" \
"     \n" \
"	const int D = 5;     \n" \
"     \n" \
"	int center = src.ptr(2 * y)[2 * x];     \n" \
"     \n" \
"	int tx = min(2 * x - D / 2 + D, src.cols - 1);     \n" \
"	int ty = min(2 * y - D / 2 + D, src.rows - 1);     \n" \
"	int cy = max(0, 2 * y - D / 2);     \n" \
"     \n" \     
"	float sum = 0;     \n" \
"	int count = 0;     \n" \
"     \n" \
"	for (; cy < ty; ++cy)     \n" \
"		for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx)     \n" \
"		{     \n" \
"			//This might not be right, but it stops incomplete model images from making up colors     \n" \
"			if (src.ptr(cy)[cx] > 0)     \n" \
"			{     \n" \
"				sum += src.ptr(cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];     \n" \
"				count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];     \n" \
"			}     \n" \
"		}     \n" \
"	dst.ptr(y)[x] = (sum / (float)count);     \n" \
"}     \n" \
"\n";

int pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, DeviceArray2D<unsigned char> & dst)
{
	/*dst.create(src.rows() / 2, src.cols() / 2);

	dim3 block(32, 8);
	dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

	const float gaussKernel[25] = { 1, 4, 6, 4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1, 4, 6, 4, 1 };

	float * gauss_cuda;

	cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
	cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

	pyrDownKernelIntensityGauss << <grid, block >> >(src, dst, gauss_cuda);
	cudaSafeCall(cudaGetLastError());

	cudaFree(gauss_cuda);*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;          
	cl_context context;               
	cl_command_queue queue;          
	cl_program program;               
	cl_kernel kernel;               

	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& pyrDownKernelIntensityGaussSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pyrDownKernelIntensityGauss", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
};


const char *verticesToDepthKernelSource = "\n" \
"__kernel void verticesToDepthKernel(const float * vmap_src,      \n" \
"									PtrStepSz<float> dst,       \n" \
"									float cutOff)      \n" \
"{      \n" \
"	int x = get_global_id(0);      \n" \
"	int y = get_global_id(1);      \n" \
"      \n" \
"	if (x >= dst.cols || y >= dst.rows)      \n" \
"		return;      \n" \
"      \n" \
"	float z = vmap_src[y * dst.cols * 4 + (x * 4) + 2];      \n" \
"      \n" \
"	dst.ptr(y)[x] = z > cutOff || z <= 0 ? __int_as_float(0x7fffffff) : z;      \n" \
"}      \n" \
"\n";


void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float> & dst, float cutOff)
{
	/*dim3 block(32, 8);
	dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

	verticesToDepthKernel << <grid, block >> >(vmap_src, dst, cutOff);
	cudaSafeCall(cudaGetLastError());*/

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& verticesToDepthKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "verticesToDepthKernele", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
};

texture<uchar4, 2, cudaReadModeElementType> inTex;


const char *bgr2IntensityKernelSource = "\n" \
"__kernel void bgr2IntensityKernel(PtrStepSz<unsigned char> dst)        \n" \
"{        \n" \
"	int x = get_global_id(0);        \n" \
"	int y = get_global_id(1);        \n" \
"        \n" \
"	if (x >= dst.cols || y >= dst.rows)        \n" \
"		return;        \n" \
"        \n" \
"	uchar4 src = tex2D(inTex, x, y);        \n" \
"        \n" \
"	int value = (float)src.x * 0.114f + (float)src.y * 0.299f + (float)src.z * 0.587f;        \n" \
"        \n" \
"	dst.ptr(y)[x] = value;        \n" \
"}        \n" \
"\n";


void imageBGRToIntensity(cudaArray * cuArr, DeviceArray2D<unsigned char> & dst)
{
	/*dim3 block(32, 8);
	dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

	cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));

	bgr2IntensityKernel << <grid, block >> >(dst);

	cudaSafeCall(cudaGetLastError());

	cudaSafeCall(cudaUnbindTexture(inTex));*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;              
	cl_command_queue queue;          
	cl_program program;               
	cl_kernel kernel;                 

	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& bgr2IntensityKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "bgr2IntensityKernel", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
};

const float gsobel_x3x3[9];
const float gsobel_y3x3[9];

const char *applyKernelSource = "\n" \
"__kernel void applyKernel(const PtrStepSz<unsigned char> src,       \n" \
"							PtrStep<short> dx,        \n" \
"							PtrStep<short> dy)       \n" \
"{       \n" \
"	int x = get_global_id(0);       \n" \
"	int y = get_global_id(1);       \n" \
"       \n" \
"	if (x >= src.cols || y >= src.rows)       \n" \
"		return;       \n" \
"       \n" \
"	float dxVal = 0;       \n" \
"	float dyVal = 0;       \n" \
"       \n" \
"	int kernelIndex = 8;       \n" \
"	for (int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++)       \n" \
"	{       \n" \
"		for (int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++)       \n" \
"		{       \n" \
"			dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];       \n" \
"			dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];       \n" \
"			--kernelIndex;       \n" \
"		}       \n" \
"	}       \n" \
"       \n" \
"	dx.ptr(y)[x] = dxVal;       \n" \
"	dy.ptr(y)[x] = dyVal;       \n" \
"}       \n" \
"\n";


int computeDerivativeImages(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy)
{
	static bool once = false;

	if (!once)
	{
		float gsx3x3[9] = { 0.52201,  0.00000, -0.52201,
			0.79451, -0.00000, -0.79451,
			0.52201,  0.00000, -0.52201 };

		float gsy3x3[9] = { 0.52201, 0.79451, 0.52201,
			0.00000, 0.00000, 0.00000,
			-0.52201, -0.79451, -0.52201 };

		clenquewritebuffer(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
		clenquewritebuffer(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

		//cudaSafeCall(cudaGetLastError());
		//cudaSafeCall(cudaDeviceSynchronize());

		once = true;
	}

	//dim3 block(32, 8);
	//dim3 grid(getGridDim(src.cols(), block.x), getGridDim(src.rows(), block.y));

	/*applyKernel << <grid, block >> >(src, dx, dy);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;              
	cl_command_queue queue;          
	cl_program program;              
	cl_kernel kernel;                 


	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	float NDRange(get_global_size(dst.cols(), block.x), get_global_size(dst.rows(), block.y));

	const float sigma_color = 30;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& applyKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "applyKernel", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
}


const char *projectPointsKernelSource = "\n" \
"__kernel void projectPointsKernel(const PtrStepSz<float> depth,        \n" \
"	PtrStepSz<float3> cloud,        \n" \
"	const float invFx,        \n" \
"	const float invFy,        \n" \
"	const float cx,        \n" \
"	const float cy)        \n" \
"{        \n" \
"	int x = get_global_id(0);        \n" \
"	int y = get_global_id(1);        \n" \
"        \n" \
"	if (x >= depth.cols || y >= depth.rows)        \n" \
"		return;        \n" \
"        \n" \
"	float z = depth.ptr(y)[x];        \n" \
"        \n" \
"	cloud.ptr(y)[x].x = (float)((x - cx) * z * invFx);        \n" \
"	cloud.ptr(y)[x].y = (float)((y - cy) * z * invFy);        \n" \
"	cloud.ptr(y)[x].z = z;        \n" \
"}        \n" \
"\n";


int projectToPointCloud(const DeviceArray2D<float> & depth,
	const DeviceArray2D<float3> & cloud,
	CameraModel & intrinsics,
	const int & level)
{
	/*dim3 block(32, 8);
	dim3 grid(getGridDim(depth.cols(), block.x), getGridDim(depth.rows(), block.y));

	CameraModel intrinsicsLevel = intrinsics(level);

	projectPointsKernel << <grid, block >> >(depth, cloud, 1.0f / intrinsicsLevel.fx, 1.0f / intrinsicsLevel.fy, intrinsicsLevel.cx, intrinsicsLevel.cy);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());*/

	cl_platform_id cpPlatform;        
	cl_device_id device_id;           
	cl_context context;              
	cl_command_queue queue;           
	cl_program program;               
	cl_kernel kernel;                 

	dst.create(src.rows() / 2, src.cols() / 2);

	// Number of work items in each local work group
	localSize = 256;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;


	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& projectPointsKernelSource, NULL, &err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "projectPointsKernel", &err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
}
