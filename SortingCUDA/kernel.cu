
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <iostream>
#include <stdio.h>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define CUDA_CALL( call )               \
{                                       \
	cudaError_t result = call;              \
	if ( cudaSuccess != result )            \
		std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

constexpr int length = 14;
int arr[length]{ 6, 8, 1, 1, 4, 2, 9, 0, 2, 2, 5, 7, 8, 7 };

__device__ int dev_arr[length];

void printArr()
{
	std::copy(std::begin(arr), std::end(arr), std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
}

/// <summary>
/// Swaps 'a' and 'b' if 'b' is less than 'a'.
/// </summary>
__forceinline __device__ void compareAndSwap(int* a, int* b)
{
	if (*b < *a)
	{
		int temp = *a;
		*a = *b;
		*b = temp;
	}
}

__global__ void sortGPUSingleBlockCUDA(int phase)
{
	int idx = threadIdx.x;
	idx *= 2;
	if (phase == 0 && idx + 1 < length)
		compareAndSwap(&dev_arr[idx], &dev_arr[idx + 1]);
	if (phase == 1 && idx + 2 < length)
		compareAndSwap(&dev_arr[idx + 1], &dev_arr[idx + 2]);
}

void sortGPUSingleBlock()
{
	CUDA_CALL(cudaMemcpyToSymbol(dev_arr, arr, sizeof(arr)));

	for (int i = 0; i < length; i++)
		sortGPUSingleBlockCUDA KERNEL_ARGS2(1, length / 2)(i % 2);

	CUDA_CALL(cudaMemcpyFromSymbol(arr, dev_arr, sizeof(arr)));
}

int main()
{
	std::cout << "Array before sorting: \n";
	printArr();

	sortGPUSingleBlock();

	std::cout << "Array after sorting: \n";
	printArr();
}
