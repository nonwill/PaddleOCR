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
#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/paddleocr.h>

namespace PaddleOCR {

struct PPOCR::PPOCR_PRIVATE
{
  Args const args;
  DBDetector * detector;
  Classifier * classifier;
  CRNNRecognizer * recognizer;

  PPOCR_PRIVATE(Args const & args_) noexcept :
    args(args_),
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

PPOCR::PPOCR(Args const & args) noexcept :
  pri(new PPOCR_PRIVATE(args))
{
  if (args.det) {
    pri->detector = new DBDetector(pri->args);
  }

  if (args.cls && args.use_angle_cls) {
    pri->classifier = new Classifier(pri->args);
  }
  if (args.rec) {
    pri->recognizer = new CRNNRecognizer(pri->args);
  }
}

PPOCR::~PPOCR()
{
  delete pri;
}

Args const & PPOCR::args() const noexcept
{
  return pri->args;
}

std::vector<std::vector<OCRPredictResult>>
PPOCR::ocr(const std::vector<cv::Mat> &img_list) noexcept {
  std::vector<std::vector<OCRPredictResult>> ocr_results;

  if (!pri->args.det) {
    std::vector<OCRPredictResult> ocr_result;
    ocr_result.resize(img_list.size());
    if (pri->args.cls && pri->classifier) {
      this->cls(img_list, ocr_result);
      for (size_t i = 0; i < img_list.size(); ++i) {
        if (ocr_result[i].cls_label % 2 == 1 &&
            ocr_result[i].cls_score > pri->args.cls_thresh) {
          cv::rotate(img_list[i], img_list[i], 1);
        }
      }
    }
    if (pri->args.rec) {
      this->rec(img_list, ocr_result);
    }
    for (size_t i = 0; i < ocr_result.size(); ++i) {
      ocr_results.emplace_back(1, std::move(ocr_result[i]));
    }
  } else {
    for (size_t i = 0; i < img_list.size(); ++i) {
      std::vector<OCRPredictResult> ocr_result =
          this->ocr(img_list[i]);
      ocr_results.emplace_back(std::move(ocr_result));
    }
  }
  return ocr_results;
}

std::vector<OCRPredictResult> PPOCR::ocr(const cv::Mat &img, bool and_cls) noexcept {

  std::vector<OCRPredictResult> ocr_result;
  // det
  this->det(img, ocr_result);
  // crop image
  std::vector<cv::Mat> img_list;
  for (size_t j = 0; j < ocr_result.size(); ++j) {
    cv::Mat crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
    img_list.emplace_back(std::move(crop_img));
  }
  // cls
  if (and_cls && pri->args.cls && pri->classifier) {
    this->cls(img_list, ocr_result);
    for (size_t i = 0; i < img_list.size(); ++i) {
      if (ocr_result[i].cls_label % 2 == 1 &&
          ocr_result[i].cls_score > pri->args.cls_thresh) {
        cv::rotate(img_list[i], img_list[i], 1);
      }
    }
  }
  // rec
  if (pri->args.rec) {
    this->rec(img_list, ocr_result);
  }
  return ocr_result;
}

void PPOCR::det(const cv::Mat &img,
                std::vector<OCRPredictResult> &ocr_results) noexcept {
  std::vector<std::vector<std::vector<int>>> boxes;

  pri->detector->Run(img, boxes);

  for (size_t i = 0; i < boxes.size(); ++i) {
    OCRPredictResult res;
    res.box = std::move(boxes[i]);
    ocr_results.emplace_back(std::move(res));
  }
  // sort boex from top to bottom, from left to right
  Utility::sorted_boxes(ocr_results);
}

void PPOCR::rec(const std::vector<cv::Mat> &img_list,
                std::vector<OCRPredictResult> &ocr_results) noexcept {
  std::vector<std::string> rec_texts(img_list.size(), std::string());
  std::vector<float> rec_text_scores(img_list.size(), 0);
  pri->recognizer->Run(img_list, rec_texts, rec_text_scores);
  // output rec results
  for (size_t i = 0; i < rec_texts.size(); ++i) {
    ocr_results[i].text = std::move(rec_texts[i]);
    ocr_results[i].score = rec_text_scores[i];
  }
}

void PPOCR::cls(const std::vector<cv::Mat> &img_list,
                std::vector<OCRPredictResult> &ocr_results) noexcept {
  std::vector<int> cls_labels(img_list.size(), 0);
  std::vector<float> cls_scores(img_list.size(), 0);
  pri->classifier->Run(img_list, cls_labels, cls_scores);
  // output cls results
  for (size_t i = 0; i < cls_labels.size(); ++i) {
    ocr_results[i].cls_label = cls_labels[i];
    ocr_results[i].cls_score = cls_scores[i];
  }
}

} // namespace PaddleOCR
