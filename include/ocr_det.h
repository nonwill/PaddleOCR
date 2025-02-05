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

#ifndef PPOCR_OCR_DET_HH
#define PPOCR_OCR_DET_HH

#include <include/postprocess_op.h>
#include <include/preprocess_op.h>
#include <memory>

namespace paddle_infer {
class Predictor;
}

namespace PaddleOCR {

class DBDetector {
public:
  explicit DBDetector(const std::string &model_dir,
                      const bool &use_gpu = false,
                      const int &gpu_id = 0, const int &gpu_mem = 4000,
                      const int &cpu_math_library_num_threads = 4,
                      const bool &use_mkldnn = false,
                      const std::string &limit_type = "max",
                      const int &limit_side_len = 960,
                      const double &det_db_thresh = 0.3,
                      const double &det_db_box_thresh = 0.5,
                      const double &det_db_unclip_ratio = 2.0,
                      const std::string &det_db_score_mode = "slow",
                      const bool &use_dilation = false,
                      const bool &use_tensorrt = false,
                      const std::string &precision = "fp32") noexcept :
    use_gpu_(use_gpu), gpu_id_(gpu_id), gpu_mem_(gpu_mem),
    cpu_math_library_num_threads_(cpu_math_library_num_threads),
    use_mkldnn_(use_mkldnn), limit_type_(limit_type),
    limit_side_len_(limit_side_len), det_db_thresh_(det_db_thresh),
    det_db_box_thresh_(det_db_box_thresh),
    det_db_unclip_ratio_(det_db_unclip_ratio),
    det_db_score_mode_(det_db_score_mode),
    use_dilation_(use_dilation), visualize_(true),
    use_tensorrt_(use_tensorrt), precision_(precision)
  {
    LoadModel(model_dir);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir) noexcept;

  // Run predictor
  void Run(const cv::Mat &img,
           std::vector<std::vector<std::vector<int>>> &boxes,
           std::vector<double> &times) noexcept;

private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;

  std::string limit_type_ = "max";
  int limit_side_len_ = 960;

  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.5;
  double det_db_unclip_ratio_ = 2.0;
  std::string det_db_score_mode_ = "slow";
  bool use_dilation_ = false;

  bool visualize_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";

  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  bool is_scale_ = true;

  // pre-process
  ResizeImgType0 resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;

  // post-process
  DBPostProcessor post_processor_;
};

} // namespace PaddleOCR

#endif // PPOCR_OCR_DET_HH
