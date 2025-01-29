// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PPOCR_ARGS_HH
#define PPOCR_ARGS_HH

#ifdef PPOCR_gflags_ENABLED

#ifdef WIN32
#   define GFLAGS_IS_A_DLL 0
#   ifdef PPOCR_LIBRARY
#       define GFLAGS_DLL_DECLARE_FLAG __declspec(dllexport)
#   else
#       define GFLAGS_DLL_DECLARE_FLAG __declspec(dllimport)
#   endif
#else
#   define GFLAGS_DLL_DECLARE_FLAG
# endif

#include <gflags/gflags.h>

#ifdef __cplusplus
extern "C" {
#endif

#else

#include <ppocr_api.h>
#include <string>
#include <ostream>

namespace PaddleOCR {

struct PPOCR_API Args {
    Args() noexcept;
    Args( int argc, char** argv ) noexcept
    { parse(argc, argv); }
    ~Args() {}

    int parse( int argc, char** argv ) noexcept;

#define DECLARE_bool(x) bool x
#define DECLARE_int32(x) int x
#define DECLARE_double(x) double x
#define DECLARE_string(x) std::string x

DECLARE_bool(help);

#endif

// common args
DECLARE_bool(use_gpu);
DECLARE_bool(use_tensorrt);
DECLARE_int32(gpu_id);
DECLARE_int32(gpu_mem);
DECLARE_int32(cpu_threads);
DECLARE_bool(enable_mkldnn);
DECLARE_string(precision);
DECLARE_bool(benchmark);
DECLARE_string(output);
DECLARE_string(image_dir);
DECLARE_string(type);
// detection related
DECLARE_string(det_model_dir);
DECLARE_string(limit_type);
DECLARE_int32(limit_side_len);
DECLARE_double(det_db_thresh);
DECLARE_double(det_db_box_thresh);
DECLARE_double(det_db_unclip_ratio);
DECLARE_bool(use_dilation);
DECLARE_string(det_db_score_mode);
DECLARE_bool(visualize);
// classification related
DECLARE_bool(use_angle_cls);
DECLARE_string(cls_model_dir);
DECLARE_double(cls_thresh);
DECLARE_int32(cls_batch_num);
// recognition related
DECLARE_string(rec_model_dir);
DECLARE_int32(rec_batch_num);
DECLARE_string(rec_char_dict_path);
DECLARE_int32(rec_img_h);
DECLARE_int32(rec_img_w);
// layout model related
DECLARE_string(layout_model_dir);
DECLARE_string(layout_dict_path);
DECLARE_double(layout_score_threshold);
DECLARE_double(layout_nms_threshold);
// structure model related
DECLARE_string(table_model_dir);
DECLARE_int32(table_max_len);
DECLARE_int32(table_batch_num);
DECLARE_string(table_char_dict_path);
DECLARE_bool(merge_no_span_structure);
// forward related
DECLARE_bool(det);
DECLARE_bool(rec);
DECLARE_bool(cls);
DECLARE_bool(table);
DECLARE_bool(layout);


#ifdef PPOCR_gflags_ENABLED
unsigned int GFLAGS_DLL_DECLARE_FLAG ppocrParseCommandLineFlags( int *argc, char*** argv, bool remove_flags );

#ifdef __cplusplus
}
#endif
#else
#undef DECLARE_bool
#undef DECLARE_int32
#undef DECLARE_double
#undef DECLARE_string
};

void PPOCR_API ArgsHelp(std::ostream & out) noexcept;

} // PaddleOCR namespace
#endif

#endif // PPOCR_ARGS_HH
