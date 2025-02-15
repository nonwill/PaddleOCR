//
// AUTOGENERATED, DO NOT EDIT
//
#ifndef OPENCV_CORE_OCL_RUNTIME_OPENCL_WRAPPERS_HPP
#error "Invalid usage"
#endif

// generated by parser_cl.py
#undef clBuildProgram
#define clBuildProgram clBuildProgram_fn
inline cl_int clBuildProgram(cl_program p0, cl_uint p1, const cl_device_id *p2,
                             const char *p3,
                             void(CL_CALLBACK *p4)(cl_program, void *),
                             void *p5) {
  return clBuildProgram_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clCompileProgram
#define clCompileProgram clCompileProgram_fn
inline cl_int clCompileProgram(cl_program p0, cl_uint p1,
                               const cl_device_id *p2, const char *p3,
                               cl_uint p4, const cl_program *p5,
                               const char **p6,
                               void(CL_CALLBACK *p7)(cl_program, void *),
                               void *p8) {
  return clCompileProgram_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clCreateBuffer
#define clCreateBuffer clCreateBuffer_fn
inline cl_mem clCreateBuffer(cl_context p0, cl_mem_flags p1, size_t p2,
                             void *p3, cl_int *p4) {
  return clCreateBuffer_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateCommandQueue
#define clCreateCommandQueue clCreateCommandQueue_fn
inline cl_command_queue clCreateCommandQueue(cl_context p0, cl_device_id p1,
                                             cl_command_queue_properties p2,
                                             cl_int *p3) {
  return clCreateCommandQueue_pfn(p0, p1, p2, p3);
}
#undef clCreateContext
#define clCreateContext clCreateContext_fn
inline cl_context clCreateContext(
    const cl_context_properties *p0, cl_uint p1, const cl_device_id *p2,
    void(CL_CALLBACK *p3)(const char *, const void *, size_t, void *), void *p4,
    cl_int *p5) {
  return clCreateContext_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clCreateContextFromType
#define clCreateContextFromType clCreateContextFromType_fn
inline cl_context clCreateContextFromType(
    const cl_context_properties *p0, cl_device_type p1,
    void(CL_CALLBACK *p2)(const char *, const void *, size_t, void *), void *p3,
    cl_int *p4) {
  return clCreateContextFromType_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateImage
#define clCreateImage clCreateImage_fn
inline cl_mem clCreateImage(cl_context p0, cl_mem_flags p1,
                            const cl_image_format *p2, const cl_image_desc *p3,
                            void *p4, cl_int *p5) {
  return clCreateImage_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clCreateImage2D
#define clCreateImage2D clCreateImage2D_fn
inline cl_mem clCreateImage2D(cl_context p0, cl_mem_flags p1,
                              const cl_image_format *p2, size_t p3, size_t p4,
                              size_t p5, void *p6, cl_int *p7) {
  return clCreateImage2D_pfn(p0, p1, p2, p3, p4, p5, p6, p7);
}
#undef clCreateImage3D
#define clCreateImage3D clCreateImage3D_fn
inline cl_mem clCreateImage3D(cl_context p0, cl_mem_flags p1,
                              const cl_image_format *p2, size_t p3, size_t p4,
                              size_t p5, size_t p6, size_t p7, void *p8,
                              cl_int *p9) {
  return clCreateImage3D_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
}
#undef clCreateKernel
#define clCreateKernel clCreateKernel_fn
inline cl_kernel clCreateKernel(cl_program p0, const char *p1, cl_int *p2) {
  return clCreateKernel_pfn(p0, p1, p2);
}
#undef clCreateKernelsInProgram
#define clCreateKernelsInProgram clCreateKernelsInProgram_fn
inline cl_int clCreateKernelsInProgram(cl_program p0, cl_uint p1, cl_kernel *p2,
                                       cl_uint *p3) {
  return clCreateKernelsInProgram_pfn(p0, p1, p2, p3);
}
#undef clCreateProgramWithBinary
#define clCreateProgramWithBinary clCreateProgramWithBinary_fn
inline cl_program clCreateProgramWithBinary(cl_context p0, cl_uint p1,
                                            const cl_device_id *p2,
                                            const size_t *p3,
                                            const unsigned char **p4,
                                            cl_int *p5, cl_int *p6) {
  return clCreateProgramWithBinary_pfn(p0, p1, p2, p3, p4, p5, p6);
}
#undef clCreateProgramWithBuiltInKernels
#define clCreateProgramWithBuiltInKernels clCreateProgramWithBuiltInKernels_fn
inline cl_program clCreateProgramWithBuiltInKernels(cl_context p0, cl_uint p1,
                                                    const cl_device_id *p2,
                                                    const char *p3,
                                                    cl_int *p4) {
  return clCreateProgramWithBuiltInKernels_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateProgramWithSource
#define clCreateProgramWithSource clCreateProgramWithSource_fn
inline cl_program clCreateProgramWithSource(cl_context p0, cl_uint p1,
                                            const char **p2, const size_t *p3,
                                            cl_int *p4) {
  return clCreateProgramWithSource_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateSampler
#define clCreateSampler clCreateSampler_fn
inline cl_sampler clCreateSampler(cl_context p0, cl_bool p1,
                                  cl_addressing_mode p2, cl_filter_mode p3,
                                  cl_int *p4) {
  return clCreateSampler_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateSubBuffer
#define clCreateSubBuffer clCreateSubBuffer_fn
inline cl_mem clCreateSubBuffer(cl_mem p0, cl_mem_flags p1,
                                cl_buffer_create_type p2, const void *p3,
                                cl_int *p4) {
  return clCreateSubBuffer_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateSubDevices
#define clCreateSubDevices clCreateSubDevices_fn
inline cl_int clCreateSubDevices(cl_device_id p0,
                                 const cl_device_partition_property *p1,
                                 cl_uint p2, cl_device_id *p3, cl_uint *p4) {
  return clCreateSubDevices_pfn(p0, p1, p2, p3, p4);
}
#undef clCreateUserEvent
#define clCreateUserEvent clCreateUserEvent_fn
inline cl_event clCreateUserEvent(cl_context p0, cl_int *p1) {
  return clCreateUserEvent_pfn(p0, p1);
}
#undef clEnqueueBarrier
#define clEnqueueBarrier clEnqueueBarrier_fn
inline cl_int clEnqueueBarrier(cl_command_queue p0) {
  return clEnqueueBarrier_pfn(p0);
}
#undef clEnqueueBarrierWithWaitList
#define clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList_fn
inline cl_int clEnqueueBarrierWithWaitList(cl_command_queue p0, cl_uint p1,
                                           const cl_event *p2, cl_event *p3) {
  return clEnqueueBarrierWithWaitList_pfn(p0, p1, p2, p3);
}
#undef clEnqueueCopyBuffer
#define clEnqueueCopyBuffer clEnqueueCopyBuffer_fn
inline cl_int clEnqueueCopyBuffer(cl_command_queue p0, cl_mem p1, cl_mem p2,
                                  size_t p3, size_t p4, size_t p5, cl_uint p6,
                                  const cl_event *p7, cl_event *p8) {
  return clEnqueueCopyBuffer_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueCopyBufferRect
#define clEnqueueCopyBufferRect clEnqueueCopyBufferRect_fn
inline cl_int clEnqueueCopyBufferRect(cl_command_queue p0, cl_mem p1, cl_mem p2,
                                      const size_t *p3, const size_t *p4,
                                      const size_t *p5, size_t p6, size_t p7,
                                      size_t p8, size_t p9, cl_uint p10,
                                      const cl_event *p11, cl_event *p12) {
  return clEnqueueCopyBufferRect_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                                     p10, p11, p12);
}
#undef clEnqueueCopyBufferToImage
#define clEnqueueCopyBufferToImage clEnqueueCopyBufferToImage_fn
inline cl_int clEnqueueCopyBufferToImage(cl_command_queue p0, cl_mem p1,
                                         cl_mem p2, size_t p3, const size_t *p4,
                                         const size_t *p5, cl_uint p6,
                                         const cl_event *p7, cl_event *p8) {
  return clEnqueueCopyBufferToImage_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueCopyImage
#define clEnqueueCopyImage clEnqueueCopyImage_fn
inline cl_int clEnqueueCopyImage(cl_command_queue p0, cl_mem p1, cl_mem p2,
                                 const size_t *p3, const size_t *p4,
                                 const size_t *p5, cl_uint p6,
                                 const cl_event *p7, cl_event *p8) {
  return clEnqueueCopyImage_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueCopyImageToBuffer
#define clEnqueueCopyImageToBuffer clEnqueueCopyImageToBuffer_fn
inline cl_int clEnqueueCopyImageToBuffer(cl_command_queue p0, cl_mem p1,
                                         cl_mem p2, const size_t *p3,
                                         const size_t *p4, size_t p5,
                                         cl_uint p6, const cl_event *p7,
                                         cl_event *p8) {
  return clEnqueueCopyImageToBuffer_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueFillBuffer
#define clEnqueueFillBuffer clEnqueueFillBuffer_fn
inline cl_int clEnqueueFillBuffer(cl_command_queue p0, cl_mem p1,
                                  const void *p2, size_t p3, size_t p4,
                                  size_t p5, cl_uint p6, const cl_event *p7,
                                  cl_event *p8) {
  return clEnqueueFillBuffer_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueFillImage
#define clEnqueueFillImage clEnqueueFillImage_fn
inline cl_int clEnqueueFillImage(cl_command_queue p0, cl_mem p1, const void *p2,
                                 const size_t *p3, const size_t *p4, cl_uint p5,
                                 const cl_event *p6, cl_event *p7) {
  return clEnqueueFillImage_pfn(p0, p1, p2, p3, p4, p5, p6, p7);
}
#undef clEnqueueMapBuffer
#define clEnqueueMapBuffer clEnqueueMapBuffer_fn
inline void *clEnqueueMapBuffer(cl_command_queue p0, cl_mem p1, cl_bool p2,
                                cl_map_flags p3, size_t p4, size_t p5,
                                cl_uint p6, const cl_event *p7, cl_event *p8,
                                cl_int *p9) {
  return clEnqueueMapBuffer_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
}
#undef clEnqueueMapImage
#define clEnqueueMapImage clEnqueueMapImage_fn
inline void *clEnqueueMapImage(cl_command_queue p0, cl_mem p1, cl_bool p2,
                               cl_map_flags p3, const size_t *p4,
                               const size_t *p5, size_t *p6, size_t *p7,
                               cl_uint p8, const cl_event *p9, cl_event *p10,
                               cl_int *p11) {
  return clEnqueueMapImage_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                               p11);
}
#undef clEnqueueMarker
#define clEnqueueMarker clEnqueueMarker_fn
inline cl_int clEnqueueMarker(cl_command_queue p0, cl_event *p1) {
  return clEnqueueMarker_pfn(p0, p1);
}
#undef clEnqueueMarkerWithWaitList
#define clEnqueueMarkerWithWaitList clEnqueueMarkerWithWaitList_fn
inline cl_int clEnqueueMarkerWithWaitList(cl_command_queue p0, cl_uint p1,
                                          const cl_event *p2, cl_event *p3) {
  return clEnqueueMarkerWithWaitList_pfn(p0, p1, p2, p3);
}
#undef clEnqueueMigrateMemObjects
#define clEnqueueMigrateMemObjects clEnqueueMigrateMemObjects_fn
inline cl_int clEnqueueMigrateMemObjects(cl_command_queue p0, cl_uint p1,
                                         const cl_mem *p2,
                                         cl_mem_migration_flags p3, cl_uint p4,
                                         const cl_event *p5, cl_event *p6) {
  return clEnqueueMigrateMemObjects_pfn(p0, p1, p2, p3, p4, p5, p6);
}
#undef clEnqueueNDRangeKernel
#define clEnqueueNDRangeKernel clEnqueueNDRangeKernel_fn
inline cl_int clEnqueueNDRangeKernel(cl_command_queue p0, cl_kernel p1,
                                     cl_uint p2, const size_t *p3,
                                     const size_t *p4, const size_t *p5,
                                     cl_uint p6, const cl_event *p7,
                                     cl_event *p8) {
  return clEnqueueNDRangeKernel_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueNativeKernel
#define clEnqueueNativeKernel clEnqueueNativeKernel_fn
inline cl_int clEnqueueNativeKernel(cl_command_queue p0,
                                    void(CL_CALLBACK *p1)(void *), void *p2,
                                    size_t p3, cl_uint p4, const cl_mem *p5,
                                    const void **p6, cl_uint p7,
                                    const cl_event *p8, cl_event *p9) {
  return clEnqueueNativeKernel_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
}
#undef clEnqueueReadBuffer
#define clEnqueueReadBuffer clEnqueueReadBuffer_fn
inline cl_int clEnqueueReadBuffer(cl_command_queue p0, cl_mem p1, cl_bool p2,
                                  size_t p3, size_t p4, void *p5, cl_uint p6,
                                  const cl_event *p7, cl_event *p8) {
  return clEnqueueReadBuffer_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueReadBufferRect
#define clEnqueueReadBufferRect clEnqueueReadBufferRect_fn
inline cl_int clEnqueueReadBufferRect(cl_command_queue p0, cl_mem p1,
                                      cl_bool p2, const size_t *p3,
                                      const size_t *p4, const size_t *p5,
                                      size_t p6, size_t p7, size_t p8,
                                      size_t p9, void *p10, cl_uint p11,
                                      const cl_event *p12, cl_event *p13) {
  return clEnqueueReadBufferRect_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                                     p10, p11, p12, p13);
}
#undef clEnqueueReadImage
#define clEnqueueReadImage clEnqueueReadImage_fn
inline cl_int clEnqueueReadImage(cl_command_queue p0, cl_mem p1, cl_bool p2,
                                 const size_t *p3, const size_t *p4, size_t p5,
                                 size_t p6, void *p7, cl_uint p8,
                                 const cl_event *p9, cl_event *p10) {
  return clEnqueueReadImage_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
}
#undef clEnqueueTask
#define clEnqueueTask clEnqueueTask_fn
inline cl_int clEnqueueTask(cl_command_queue p0, cl_kernel p1, cl_uint p2,
                            const cl_event *p3, cl_event *p4) {
  return clEnqueueTask_pfn(p0, p1, p2, p3, p4);
}
#undef clEnqueueUnmapMemObject
#define clEnqueueUnmapMemObject clEnqueueUnmapMemObject_fn
inline cl_int clEnqueueUnmapMemObject(cl_command_queue p0, cl_mem p1, void *p2,
                                      cl_uint p3, const cl_event *p4,
                                      cl_event *p5) {
  return clEnqueueUnmapMemObject_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clEnqueueWaitForEvents
#define clEnqueueWaitForEvents clEnqueueWaitForEvents_fn
inline cl_int clEnqueueWaitForEvents(cl_command_queue p0, cl_uint p1,
                                     const cl_event *p2) {
  return clEnqueueWaitForEvents_pfn(p0, p1, p2);
}
#undef clEnqueueWriteBuffer
#define clEnqueueWriteBuffer clEnqueueWriteBuffer_fn
inline cl_int clEnqueueWriteBuffer(cl_command_queue p0, cl_mem p1, cl_bool p2,
                                   size_t p3, size_t p4, const void *p5,
                                   cl_uint p6, const cl_event *p7,
                                   cl_event *p8) {
  return clEnqueueWriteBuffer_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clEnqueueWriteBufferRect
#define clEnqueueWriteBufferRect clEnqueueWriteBufferRect_fn
inline cl_int clEnqueueWriteBufferRect(cl_command_queue p0, cl_mem p1,
                                       cl_bool p2, const size_t *p3,
                                       const size_t *p4, const size_t *p5,
                                       size_t p6, size_t p7, size_t p8,
                                       size_t p9, const void *p10, cl_uint p11,
                                       const cl_event *p12, cl_event *p13) {
  return clEnqueueWriteBufferRect_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                                      p10, p11, p12, p13);
}
#undef clEnqueueWriteImage
#define clEnqueueWriteImage clEnqueueWriteImage_fn
inline cl_int clEnqueueWriteImage(cl_command_queue p0, cl_mem p1, cl_bool p2,
                                  const size_t *p3, const size_t *p4, size_t p5,
                                  size_t p6, const void *p7, cl_uint p8,
                                  const cl_event *p9, cl_event *p10) {
  return clEnqueueWriteImage_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
}
#undef clFinish
#define clFinish clFinish_fn
inline cl_int clFinish(cl_command_queue p0) { return clFinish_pfn(p0); }
#undef clFlush
#define clFlush clFlush_fn
inline cl_int clFlush(cl_command_queue p0) { return clFlush_pfn(p0); }
#undef clGetCommandQueueInfo
#define clGetCommandQueueInfo clGetCommandQueueInfo_fn
inline cl_int clGetCommandQueueInfo(cl_command_queue p0,
                                    cl_command_queue_info p1, size_t p2,
                                    void *p3, size_t *p4) {
  return clGetCommandQueueInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetContextInfo
#define clGetContextInfo clGetContextInfo_fn
inline cl_int clGetContextInfo(cl_context p0, cl_context_info p1, size_t p2,
                               void *p3, size_t *p4) {
  return clGetContextInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetDeviceIDs
#define clGetDeviceIDs clGetDeviceIDs_fn
inline cl_int clGetDeviceIDs(cl_platform_id p0, cl_device_type p1, cl_uint p2,
                             cl_device_id *p3, cl_uint *p4) {
  return clGetDeviceIDs_pfn(p0, p1, p2, p3, p4);
}
#undef clGetDeviceInfo
#define clGetDeviceInfo clGetDeviceInfo_fn
inline cl_int clGetDeviceInfo(cl_device_id p0, cl_device_info p1, size_t p2,
                              void *p3, size_t *p4) {
  return clGetDeviceInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetEventInfo
#define clGetEventInfo clGetEventInfo_fn
inline cl_int clGetEventInfo(cl_event p0, cl_event_info p1, size_t p2, void *p3,
                             size_t *p4) {
  return clGetEventInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetEventProfilingInfo
#define clGetEventProfilingInfo clGetEventProfilingInfo_fn
inline cl_int clGetEventProfilingInfo(cl_event p0, cl_profiling_info p1,
                                      size_t p2, void *p3, size_t *p4) {
  return clGetEventProfilingInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetExtensionFunctionAddress
#define clGetExtensionFunctionAddress clGetExtensionFunctionAddress_fn
inline void *clGetExtensionFunctionAddress(const char *p0) {
  return clGetExtensionFunctionAddress_pfn(p0);
}
#undef clGetExtensionFunctionAddressForPlatform
#define clGetExtensionFunctionAddressForPlatform                               \
  clGetExtensionFunctionAddressForPlatform_fn
inline void *clGetExtensionFunctionAddressForPlatform(cl_platform_id p0,
                                                      const char *p1) {
  return clGetExtensionFunctionAddressForPlatform_pfn(p0, p1);
}
#undef clGetImageInfo
#define clGetImageInfo clGetImageInfo_fn
inline cl_int clGetImageInfo(cl_mem p0, cl_image_info p1, size_t p2, void *p3,
                             size_t *p4) {
  return clGetImageInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetKernelArgInfo
#define clGetKernelArgInfo clGetKernelArgInfo_fn
inline cl_int clGetKernelArgInfo(cl_kernel p0, cl_uint p1,
                                 cl_kernel_arg_info p2, size_t p3, void *p4,
                                 size_t *p5) {
  return clGetKernelArgInfo_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clGetKernelInfo
#define clGetKernelInfo clGetKernelInfo_fn
inline cl_int clGetKernelInfo(cl_kernel p0, cl_kernel_info p1, size_t p2,
                              void *p3, size_t *p4) {
  return clGetKernelInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetKernelWorkGroupInfo
#define clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo_fn
inline cl_int clGetKernelWorkGroupInfo(cl_kernel p0, cl_device_id p1,
                                       cl_kernel_work_group_info p2, size_t p3,
                                       void *p4, size_t *p5) {
  return clGetKernelWorkGroupInfo_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clGetMemObjectInfo
#define clGetMemObjectInfo clGetMemObjectInfo_fn
inline cl_int clGetMemObjectInfo(cl_mem p0, cl_mem_info p1, size_t p2, void *p3,
                                 size_t *p4) {
  return clGetMemObjectInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetPlatformIDs
#define clGetPlatformIDs clGetPlatformIDs_fn
inline cl_int clGetPlatformIDs(cl_uint p0, cl_platform_id *p1, cl_uint *p2) {
  return clGetPlatformIDs_pfn(p0, p1, p2);
}
#undef clGetPlatformInfo
#define clGetPlatformInfo clGetPlatformInfo_fn
inline cl_int clGetPlatformInfo(cl_platform_id p0, cl_platform_info p1,
                                size_t p2, void *p3, size_t *p4) {
  return clGetPlatformInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetProgramBuildInfo
#define clGetProgramBuildInfo clGetProgramBuildInfo_fn
inline cl_int clGetProgramBuildInfo(cl_program p0, cl_device_id p1,
                                    cl_program_build_info p2, size_t p3,
                                    void *p4, size_t *p5) {
  return clGetProgramBuildInfo_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clGetProgramInfo
#define clGetProgramInfo clGetProgramInfo_fn
inline cl_int clGetProgramInfo(cl_program p0, cl_program_info p1, size_t p2,
                               void *p3, size_t *p4) {
  return clGetProgramInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetSamplerInfo
#define clGetSamplerInfo clGetSamplerInfo_fn
inline cl_int clGetSamplerInfo(cl_sampler p0, cl_sampler_info p1, size_t p2,
                               void *p3, size_t *p4) {
  return clGetSamplerInfo_pfn(p0, p1, p2, p3, p4);
}
#undef clGetSupportedImageFormats
#define clGetSupportedImageFormats clGetSupportedImageFormats_fn
inline cl_int clGetSupportedImageFormats(cl_context p0, cl_mem_flags p1,
                                         cl_mem_object_type p2, cl_uint p3,
                                         cl_image_format *p4, cl_uint *p5) {
  return clGetSupportedImageFormats_pfn(p0, p1, p2, p3, p4, p5);
}
#undef clLinkProgram
#define clLinkProgram clLinkProgram_fn
inline cl_program clLinkProgram(cl_context p0, cl_uint p1,
                                const cl_device_id *p2, const char *p3,
                                cl_uint p4, const cl_program *p5,
                                void(CL_CALLBACK *p6)(cl_program, void *),
                                void *p7, cl_int *p8) {
  return clLinkProgram_pfn(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}
#undef clReleaseCommandQueue
#define clReleaseCommandQueue clReleaseCommandQueue_fn
inline cl_int clReleaseCommandQueue(cl_command_queue p0) {
  return clReleaseCommandQueue_pfn(p0);
}
#undef clReleaseContext
#define clReleaseContext clReleaseContext_fn
inline cl_int clReleaseContext(cl_context p0) {
  return clReleaseContext_pfn(p0);
}
#undef clReleaseDevice
#define clReleaseDevice clReleaseDevice_fn
inline cl_int clReleaseDevice(cl_device_id p0) {
  return clReleaseDevice_pfn(p0);
}
#undef clReleaseEvent
#define clReleaseEvent clReleaseEvent_fn
inline cl_int clReleaseEvent(cl_event p0) { return clReleaseEvent_pfn(p0); }
#undef clReleaseKernel
#define clReleaseKernel clReleaseKernel_fn
inline cl_int clReleaseKernel(cl_kernel p0) { return clReleaseKernel_pfn(p0); }
#undef clReleaseMemObject
#define clReleaseMemObject clReleaseMemObject_fn
inline cl_int clReleaseMemObject(cl_mem p0) {
  return clReleaseMemObject_pfn(p0);
}
#undef clReleaseProgram
#define clReleaseProgram clReleaseProgram_fn
inline cl_int clReleaseProgram(cl_program p0) {
  return clReleaseProgram_pfn(p0);
}
#undef clReleaseSampler
#define clReleaseSampler clReleaseSampler_fn
inline cl_int clReleaseSampler(cl_sampler p0) {
  return clReleaseSampler_pfn(p0);
}
#undef clRetainCommandQueue
#define clRetainCommandQueue clRetainCommandQueue_fn
inline cl_int clRetainCommandQueue(cl_command_queue p0) {
  return clRetainCommandQueue_pfn(p0);
}
#undef clRetainContext
#define clRetainContext clRetainContext_fn
inline cl_int clRetainContext(cl_context p0) { return clRetainContext_pfn(p0); }
#undef clRetainDevice
#define clRetainDevice clRetainDevice_fn
inline cl_int clRetainDevice(cl_device_id p0) { return clRetainDevice_pfn(p0); }
#undef clRetainEvent
#define clRetainEvent clRetainEvent_fn
inline cl_int clRetainEvent(cl_event p0) { return clRetainEvent_pfn(p0); }
#undef clRetainKernel
#define clRetainKernel clRetainKernel_fn
inline cl_int clRetainKernel(cl_kernel p0) { return clRetainKernel_pfn(p0); }
#undef clRetainMemObject
#define clRetainMemObject clRetainMemObject_fn
inline cl_int clRetainMemObject(cl_mem p0) { return clRetainMemObject_pfn(p0); }
#undef clRetainProgram
#define clRetainProgram clRetainProgram_fn
inline cl_int clRetainProgram(cl_program p0) { return clRetainProgram_pfn(p0); }
#undef clRetainSampler
#define clRetainSampler clRetainSampler_fn
inline cl_int clRetainSampler(cl_sampler p0) { return clRetainSampler_pfn(p0); }
#undef clSetEventCallback
#define clSetEventCallback clSetEventCallback_fn
inline cl_int
clSetEventCallback(cl_event p0, cl_int p1,
                   void(CL_CALLBACK *p2)(cl_event, cl_int, void *), void *p3) {
  return clSetEventCallback_pfn(p0, p1, p2, p3);
}
#undef clSetKernelArg
#define clSetKernelArg clSetKernelArg_fn
inline cl_int clSetKernelArg(cl_kernel p0, cl_uint p1, size_t p2,
                             const void *p3) {
  return clSetKernelArg_pfn(p0, p1, p2, p3);
}
#undef clSetMemObjectDestructorCallback
#define clSetMemObjectDestructorCallback clSetMemObjectDestructorCallback_fn
inline cl_int clSetMemObjectDestructorCallback(
    cl_mem p0, void(CL_CALLBACK *p1)(cl_mem, void *), void *p2) {
  return clSetMemObjectDestructorCallback_pfn(p0, p1, p2);
}
#undef clSetUserEventStatus
#define clSetUserEventStatus clSetUserEventStatus_fn
inline cl_int clSetUserEventStatus(cl_event p0, cl_int p1) {
  return clSetUserEventStatus_pfn(p0, p1);
}
#undef clUnloadCompiler
#define clUnloadCompiler clUnloadCompiler_fn
inline cl_int clUnloadCompiler() { return clUnloadCompiler_pfn(); }
#undef clUnloadPlatformCompiler
#define clUnloadPlatformCompiler clUnloadPlatformCompiler_fn
inline cl_int clUnloadPlatformCompiler(cl_platform_id p0) {
  return clUnloadPlatformCompiler_pfn(p0);
}
#undef clWaitForEvents
#define clWaitForEvents clWaitForEvents_fn
inline cl_int clWaitForEvents(cl_uint p0, const cl_event *p1) {
  return clWaitForEvents_pfn(p0, p1);
}
