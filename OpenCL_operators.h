

#include <vector_functions.h>

#ifndef CUDA_OPERATORS_CUH_
#define CUDA_OPERATORS_CUH_

__device__ __host__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ __forceinline__ float3 cross(const float3& a, const float3& b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __host__ __forceinline__ float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ __forceinline__ float norm(const float3& a)
{
	return sqrtf(dot(a, a));
}

__device__ __host__ __forceinline__ float3 normalized(const float3& a)
{
	const float rn = rsqrtf(dot(a, a));
	return make_float3(a.x * rn, a.y * rn, a.z * rn);
}

__device__ __forceinline__ float3 operator*(const mat33& m, const float3& a)
{
	return make_float3(dot(m.data[0], a), dot(m.data[1], a), dot(m.data[2], a));
}

#endif /* CUDA_OPERATORS_CUH_ */
