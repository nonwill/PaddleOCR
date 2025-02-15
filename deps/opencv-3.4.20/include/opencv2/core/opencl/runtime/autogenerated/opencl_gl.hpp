//
// AUTOGENERATED, DO NOT EDIT
//
#ifndef OPENCV_CORE_OCL_RUNTIME_OPENCL_GL_HPP
#error "Invalid usage"
#endif

// generated by parser_cl.py
#define clCreateFromGLBuffer clCreateFromGLBuffer_
#define clCreateFromGLRenderbuffer clCreateFromGLRenderbuffer_
#define clCreateFromGLTexture clCreateFromGLTexture_
#define clCreateFromGLTexture2D clCreateFromGLTexture2D_
#define clCreateFromGLTexture3D clCreateFromGLTexture3D_
#define clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects_
#define clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects_
#define clGetGLContextInfoKHR clGetGLContextInfoKHR_
#define clGetGLObjectInfo clGetGLObjectInfo_
#define clGetGLTextureInfo clGetGLTextureInfo_

#if defined __APPLE__
#include <OpenCL/cl_gl.h>
#else
#include <CL/cl_gl.h>
#endif

// generated by parser_cl.py
#undef clCreateFromGLBuffer
#define clCreateFromGLBuffer clCreateFromGLBuffer_pfn
#undef clCreateFromGLRenderbuffer
#define clCreateFromGLRenderbuffer clCreateFromGLRenderbuffer_pfn
#undef clCreateFromGLTexture
#define clCreateFromGLTexture clCreateFromGLTexture_pfn
#undef clCreateFromGLTexture2D
#define clCreateFromGLTexture2D clCreateFromGLTexture2D_pfn
#undef clCreateFromGLTexture3D
#define clCreateFromGLTexture3D clCreateFromGLTexture3D_pfn
#undef clEnqueueAcquireGLObjects
#define clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects_pfn
#undef clEnqueueReleaseGLObjects
#define clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects_pfn
#undef clGetGLContextInfoKHR
#define clGetGLContextInfoKHR clGetGLContextInfoKHR_pfn
#undef clGetGLObjectInfo
#define clGetGLObjectInfo clGetGLObjectInfo_pfn
#undef clGetGLTextureInfo
#define clGetGLTextureInfo clGetGLTextureInfo_pfn

#ifdef cl_khr_gl_sharing

// generated by parser_cl.py
extern CL_RUNTIME_EXPORT cl_mem(CL_API_CALL *clCreateFromGLBuffer)(cl_context,
                                                                   cl_mem_flags,
                                                                   cl_GLuint,
                                                                   int *);
extern CL_RUNTIME_EXPORT
cl_mem(CL_API_CALL *clCreateFromGLRenderbuffer)(cl_context, cl_mem_flags,
                                                cl_GLuint, cl_int *);
extern CL_RUNTIME_EXPORT
cl_mem(CL_API_CALL *clCreateFromGLTexture)(cl_context, cl_mem_flags, cl_GLenum,
                                           cl_GLint, cl_GLuint, cl_int *);
extern CL_RUNTIME_EXPORT cl_mem(CL_API_CALL *clCreateFromGLTexture2D)(
    cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);
extern CL_RUNTIME_EXPORT cl_mem(CL_API_CALL *clCreateFromGLTexture3D)(
    cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *);
extern CL_RUNTIME_EXPORT
cl_int(CL_API_CALL *clEnqueueAcquireGLObjects)(cl_command_queue, cl_uint,
                                               const cl_mem *, cl_uint,
                                               const cl_event *, cl_event *);
extern CL_RUNTIME_EXPORT
cl_int(CL_API_CALL *clEnqueueReleaseGLObjects)(cl_command_queue, cl_uint,
                                               const cl_mem *, cl_uint,
                                               const cl_event *, cl_event *);
extern CL_RUNTIME_EXPORT
cl_int(CL_API_CALL *clGetGLContextInfoKHR)(const cl_context_properties *,
                                           cl_gl_context_info, size_t, void *,
                                           size_t *);
extern CL_RUNTIME_EXPORT
cl_int(CL_API_CALL *clGetGLObjectInfo)(cl_mem, cl_gl_object_type *,
                                       cl_GLuint *);
extern CL_RUNTIME_EXPORT
cl_int(CL_API_CALL *clGetGLTextureInfo)(cl_mem, cl_gl_texture_info, size_t,
                                        void *, size_t *);

#endif // cl_khr_gl_sharing
