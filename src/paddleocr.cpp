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

#include <include/paddleocr.h>
#include <include/ocr_det.h>
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/args.h>

#ifdef PPOCR_benchmark_ENABLED
#include "auto_log/autolog.h"
#endif

namespace PaddleOCR {

struct PPOCR::PPOCR_PRIVATE
{
  DBDetector * detector;
  Classifier * classifier;
  CRNNRecognizer * recognizer;

  PPOCR_PRIVATE() noexcept :
    detector(nullptr),
    classifier(nullptr),
    recognizer(nullptr)
  {}

  ~PPOCR_PRIVATE()
  {
    if ( recognizer )
      delete recognizer;

    if ( classifier )
      delete classifier;

    if ( detector )
      delete detector;
  }
};

#ifdef PPOCR_gflags_ENABLED
PPOCR::PPOCR() : pri(new PPOCR_PRIVATE) {
  if (FLAGS_det) {
    pri->detector = new DBDetector(
        FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_limit_type,
        FLAGS_limit_side_len, FLAGS_det_db_thresh, FLAGS_det_db_box_thresh,
        FLAGS_det_db_unclip_ratio, FLAGS_det_db_score_mode, FLAGS_use_dilation,
        FLAGS_use_tensorrt, FLAGS_precision);
  }

  if (FLAGS_cls && FLAGS_use_angle_cls) {
    pri->classifier = new Classifier(
        FLAGS_cls_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_cls_thresh,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_cls_batch_num);
  }
  if (FLAGS_rec) {
    pri->recognizer = new CRNNRecognizer(
        FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_rec_char_dict_path,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_rec_batch_num,
        FLAGS_rec_img_h, FLAGS_rec_img_w);
  }
}
#else
PPOCR::PPOCR(Args const & args) noexcept :
  pri(new PPOCR_PRIVATE)
{
  if (args.det) {
    pri->detector = new DBDetector(
        args.det_model_dir, args.use_gpu, args.gpu_id, args.gpu_mem,
        args.cpu_threads, args.enable_mkldnn, args.limit_type,
        args.limit_side_len, args.det_db_thresh, args.det_db_box_thresh,
        args.det_db_unclip_ratio, args.det_db_score_mode, args.use_dilation,
        args.use_tensorrt, args.precision);
  }

  if (args.cls && args.use_angle_cls) {
    pri->classifier = new Classifier(
        args.cls_model_dir, args.use_gpu, args.gpu_id, args.gpu_mem,
        args.cpu_threads, args.enable_mkldnn, args.cls_thresh,
        args.use_tensorrt, args.precision, args.cls_batch_num);
  }
  if (args.rec) {
    pri->recognizer = new CRNNRecognizer(
        args.rec_model_dir, args.rec_char_dict_path,
        args.use_gpu, args.gpu_id,
        args.gpu_mem, args.cpu_threads, args.enable_mkldnn,
        args.use_tensorrt, args.precision, args.rec_batch_num,
        args.rec_img_h, args.rec_img_w);
  }
}
#endif

PPOCR::~PPOCR()
{
  delete pri;
}

std::vector<std::vector<OCRPredictResult>>
PPOCR::ocr(const std::vector<cv::Mat> &img_list, bool det, bool rec, bool cls) noexcept
{
  std::vector<std::vector<OCRPredictResult>> ocr_results;

  if (!det) {
    std::vector<OCRPredictResult> ocr_result;
    ocr_result.resize(img_list.size());
    if (cls && pri->classifier) {
      this->cls(img_list, ocr_result);
      for (int i = 0; i < img_list.size(); ++i) {
        if (ocr_result[i].cls_label % 2 == 1 &&
            ocr_result[i].cls_score > pri->classifier->cls_thresh_val()) {
          cv::rotate(img_list[i], img_list[i], 1);
        }
      }
    }
    if (rec) {
      this->rec(img_list, ocr_result);
    }
    for (int i = 0; i < ocr_result.size(); ++i) {
      std::vector<OCRPredictResult> ocr_result_tmp;
      ocr_result_tmp.emplace_back(ocr_result[i]);
      ocr_results.emplace_back(std::move(ocr_result_tmp));
    }
  } else {
    for (int i = 0; i < img_list.size(); ++i) {
      std::vector<OCRPredictResult> ocr_result =
          this->ocr(img_list[i], true, rec, cls);
      ocr_results.emplace_back(std::move(ocr_result));
    }
  }
  return ocr_results;
}

std::vector<OCRPredictResult> PPOCR::ocr(const cv::Mat &img, bool det, bool rec,
                                         bool cls) noexcept
{

  std::vector<OCRPredictResult> ocr_result;
  // det
  this->det(img, ocr_result);
  // crop image
  std::vector<cv::Mat> img_list;
  for (int j = 0; j < ocr_result.size(); j++) {
    cv::Mat crop_img;
    crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
    img_list.emplace_back(crop_img);
  }
  // cls
  if (cls && pri->classifier) {
    this->cls(img_list, ocr_result);
    for (int i = 0; i < img_list.size(); ++i) {
      if (ocr_result[i].cls_label % 2 == 1 &&
          ocr_result[i].cls_score > pri->classifier->cls_thresh_val()) {
        cv::rotate(img_list[i], img_list[i], 1);
      }
    }
  }
  // rec
  if (rec) {
    this->rec(img_list, ocr_result);
  }
  return ocr_result;
}

void PPOCR::det(const cv::Mat &img, std::vector<OCRPredictResult> &ocr_results) noexcept
{
  std::vector<std::vector<std::vector<int>>> boxes;
  std::vector<double> det_times;

  pri->detector->Run(img, boxes, det_times);

  for (int i = 0; i < boxes.size(); ++i) {
    OCRPredictResult res;
    res.box = boxes[i];
    ocr_results.emplace_back(res);
  }
  // sort boex from top to bottom, from left to right
  Utility::sorted_boxes(ocr_results);
#ifdef PPOCR_benchmark_ENABLED
  this->time_info_det[0] += det_times[0];
  this->time_info_det[1] += det_times[1];
  this->time_info_det[2] += det_times[2];
#endif
}

void PPOCR::rec(const std::vector<cv::Mat> &img_list,
                std::vector<OCRPredictResult> &ocr_results) noexcept
{
  std::vector<std::string> rec_texts(img_list.size(), "");
  std::vector<float> rec_text_scores(img_list.size(), 0);
  std::vector<double> rec_times;
  pri->recognizer->Run(img_list, rec_texts, rec_text_scores, rec_times);
  // output rec results
  for (int i = 0; i < rec_texts.size(); ++i) {
    ocr_results[i].text = rec_texts[i];
    ocr_results[i].score = rec_text_scores[i];
  }
#ifdef PPOCR_benchmark_ENABLED
  this->time_info_rec[0] += rec_times[0];
  this->time_info_rec[1] += rec_times[1];
  this->time_info_rec[2] += rec_times[2];
#endif
}

void PPOCR::cls(const std::vector<cv::Mat> &img_list,
                std::vector<OCRPredictResult> &ocr_results) noexcept
{
  std::vector<int> cls_labels(img_list.size(), 0);
  std::vector<float> cls_scores(img_list.size(), 0);
  std::vector<double> cls_times;
  pri->classifier->Run(img_list, cls_labels, cls_scores, cls_times);
  // output cls results
  for (int i = 0; i < cls_labels.size(); ++i) {
    ocr_results[i].cls_label = cls_labels[i];
    ocr_results[i].cls_score = cls_scores[i];
  }
#ifdef PPOCR_benchmark_ENABLED
  this->time_info_cls[0] += cls_times[0];
  this->time_info_cls[1] += cls_times[1];
  this->time_info_cls[2] += cls_times[2];
#endif
}

#ifdef PPOCR_benchmark_ENABLED
void PPOCR::reset_timer() {
  this->time_info_det = {0, 0, 0};
  this->time_info_rec = {0, 0, 0};
  this->time_info_cls = {0, 0, 0};
}

void PPOCR::benchmark_log(int img_num) {
  if (this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2] >
      0) {
    AutoLogger autolog_det("ocr_det", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads, 1, "dynamic",
                           FLAGS_precision, this->time_info_det, img_num);
    autolog_det.report();
  }
  if (this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2] >
      0) {
    AutoLogger autolog_rec("ocr_rec", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_rec_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_rec, img_num);
    autolog_rec.report();
  }
  if (this->time_info_cls[0] + this->time_info_cls[1] + this->time_info_cls[2] >
      0) {
    AutoLogger autolog_cls("ocr_cls", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_cls_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_cls, img_num);
    autolog_cls.report();
  }
}
#endif

} // namespace PaddleOCR
