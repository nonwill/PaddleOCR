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

#include <include/args.h>
#include <include/preprocess_op.h>
#include <include/structure_layout.h>
#include <paddle_inference_api.h>

#include <iostream>
#include <numeric>

namespace PaddleOCR {
StructureLayoutRecognizer::StructureLayoutRecognizer(Args const &args) noexcept
    : args_(args), mean_({0.485f, 0.456f, 0.406f}),
      scale_({1 / 0.229f, 1 / 0.224f, 1 / 0.225f}), is_scale_(true),
      post_processor_(args.layout_dict_path, args.layout_score_threshold,
                      args.layout_nms_threshold) {
  LoadModel(args.layout_model_dir);
}

void StructureLayoutRecognizer::Run(
    const cv::Mat &img, std::vector<StructurePredictResult> &result) noexcept {
  // preprocess

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat resize_img;
  Resize::Run(srcimg, resize_img, 800, 608);
  Normalize::Run(resize_img, this->mean_, this->scale_, this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  Permute::Run(resize_img, input.data());

  // inference.
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  input_t->CopyFromCpu(input.data());

  this->predictor_->Run();

  // Get output tensor
  std::vector<std::vector<float>> out_tensor_list;
  std::vector<std::vector<int>> output_shape_list;
  auto output_names = this->predictor_->GetOutputNames();
  for (size_t j = 0; j < output_names.size(); ++j) {
    auto output_tensor = this->predictor_->GetOutputHandle(output_names[j]);
    std::vector<int> output_shape = output_tensor->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    output_shape_list.emplace_back(std::move(output_shape));

    std::vector<float> out_data;
    out_data.resize(out_num);
    output_tensor->CopyToCpu(out_data.data());
    out_tensor_list.emplace_back(std::move(out_data));
  }

  // postprocess

  std::vector<int> bbox_num;
  int reg_max = 0;
  for (size_t i = 0; i < out_tensor_list.size(); ++i) {
    if (i == this->post_processor_.fpn_stride_size()) {
      reg_max = output_shape_list[i][2] / 4;
      break;
    }
  }
  std::vector<int> ori_shape = {srcimg.rows, srcimg.cols};
  std::vector<int> resize_shape = {resize_img.rows, resize_img.cols};
  this->post_processor_.Run(result, out_tensor_list, ori_shape, resize_shape,
                            reg_max);
  bbox_num.emplace_back(result.size());
}

void StructureLayoutRecognizer::LoadModel(
    const std::string &model_dir) noexcept {
  paddle_infer::Config config;
  bool json_model = false;
  std::string model_file_path, param_file_path;
  char const *model_variants[8] = {"/inference.json",    "/inference.pdiparams",
                                   "/model.json",        "/model.pdiparams",
                                   "/inference.pdmodel", "/inference.pdiparams",
                                   "/model.pdmodel",     "/model.pdiparams"};
  for (int i = 0; i < 8; i += 2) {
    std::string model_file = model_dir + model_variants[i];
    if (Utility::PathExists(model_file)) {
      model_file_path = std::move(model_file);
      param_file_path = model_dir + model_variants[i + 1];
      json_model = (i == 0 || i == 2);
      break;
    }
  }
  if (model_file_path.empty()) {
    fprintf(stderr, "[ERROR] No valid model file found in %s\n",
            model_dir.c_str());
    fflush(stderr);
    return;
  }
  config.SetModel(model_file_path, param_file_path);

  if (args_.use_gpu) {
    config.EnableUseGpu(args_.gpu_mem, args_.gpu_id);
    if (args_.use_tensorrt) {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (args_.precision == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      if (args_.precision == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      }
      config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
      if (!Utility::PathExists("./trt_layout_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_layout_shape.txt");
      } else {
        config.EnableTunedTensorRtDynamicShape("./trt_layout_shape.txt", true);
      }
    }
  } else {
    config.DisableGpu();
    if (args_.enable_mkldnn) {
      config.EnableMKLDNN();
    } else {
      config.DisableMKLDNN();
    }
    config.SetCpuMathLibraryNumThreads(args_.cpu_threads);
    if (json_model) {
      config.EnableNewIR();
      config.EnableNewExecutor();
    }
  }

  // false for zero copy tensor
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = paddle_infer::CreatePredictor(config);
}

} // namespace PaddleOCR
