/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_NVIDIA_CUDA_HIP_COMPAT_HPP
#define MEEP_BACKEND_NVIDIA_CUDA_HIP_COMPAT_HPP

/* Keep the existing CUDA implementation as the source of truth.  The private
   HIP portability target defines MEEP_HIP_PORTABILITY and compiles the same
   translation units through these spelling aliases. */
#if defined(MEEP_HIP_PORTABILITY)

#include <hip/hip_math_constants.h>
#include <hip/hip_runtime.h>

#define cudaDevAttrMaxGridDimX hipDeviceAttributeMaxGridDimX
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorStreamCaptureInvalidated hipErrorStreamCaptureInvalidated
#define cudaErrorUnknown hipErrorUnknown
#define cudaError_t hipError_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphCreate hipGraphCreate
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaGraphExecUpdateError hipGraphExecUpdateError
#define cudaGraphExecUpdateResult hipGraphExecUpdateResult
#define cudaGraphExecUpdateSuccess hipGraphExecUpdateSuccess
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphNode_t hipGraphNode_t
#define cudaGraph_t hipGraph_t
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaMalloc hipMalloc
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaUUID_t hipUUID

#define MEEP_DEVICE_INFINITY HIP_INF
#define MEEP_POINTER_MEMORY_TYPE(attributes) ((attributes).type)
#define MEEP_GRAPH_EXEC_UPDATE_AVAILABLE 1
#define MEEP_GRAPH_RUNTIME_CAPABILITY_ENABLED 0

#else

#include <cuda_runtime.h>
#include <math_constants.h>

#define MEEP_DEVICE_INFINITY CUDART_INF
#if CUDART_VERSION >= 10000
#define MEEP_POINTER_MEMORY_TYPE(attributes) ((attributes).type)
#else
#define MEEP_POINTER_MEMORY_TYPE(attributes) ((attributes).memoryType)
#endif
#if CUDART_VERSION >= 10020
#define MEEP_GRAPH_EXEC_UPDATE_AVAILABLE 1
#else
#define MEEP_GRAPH_EXEC_UPDATE_AVAILABLE 0
#endif
#define MEEP_GRAPH_RUNTIME_CAPABILITY_ENABLED 1

#endif

#endif
