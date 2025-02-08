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
#include <ppocr_c.h>
#include <paddlestructure.h>
#include <opencv2/imgcodecs.hpp>
#include <args.h>
#include <iostream>

using namespace PaddleOCR;

int check_params( Args const & args ) noexcept {
  if (args.det) {
    if (args.det_model_dir.empty() || args.image_dir.empty()) {
      return 1;
    }
  }

  if (args.rec) {
    if (args.rec_model_dir.empty() || args.image_dir.empty()) {
      return 2;
    }
  }

  if (args.cls && args.use_angle_cls) {
    if (args.cls_model_dir.empty() || args.image_dir.empty()) {
      return 3;
    }
  }

  if (args.table) {
    if (args.table_model_dir.empty() || args.det_model_dir.empty() ||
        args.rec_model_dir.empty() || args.image_dir.empty()) {
       return 4;
    }
  }

  if (args.layout) {
    if (args.layout_model_dir.empty() || args.image_dir.empty()) {
      return 5;
    }
  }

  if (args.precision != "fp32" && args.precision != "fp16" &&
      args.precision != "int8") {
    return 6;
  }

  return 0;
}

/// Print help messages
void ppocr_print_help() noexcept
{
  ArgsHelp(std::cout);
}

struct PPOCRC {
  PPOCR* exe;
};

int ppocr_from_Args( CPPOCR * cppocr, Args const & args ) noexcept
{
  if ( args.help )
  {
    ppocr_print_help();
    exit(0);
  }

  int ret = check_params(args);
  if ( ret )
    return ret;

  PPOCR * ocr;

  if (args.type == 1)
    ocr = new PaddleOCR::PPOCR(args);
  else if (args.type == 2)
    ocr = new PaddleOCR::PaddleStructure(args);
  else
    return 7;

  if ( !ocr )
    return -2;

  *cppocr = new struct PPOCRC;
  if ( !(*cppocr) )
  {
    delete ocr;
    return -1;
  }

  (*cppocr)->exe = ocr;

  return 0;
}


int ppocr_from_args( CPPOCR * cppocr, int argc, char ** argv ) noexcept
{
  return ppocr_from_Args( cppocr, Args( argc, argv ) );
}

int ppocr_from_inis( CPPOCR * cppocr, const char *inis ) noexcept
{
  return ppocr_from_Args( cppocr, Args(inis) );
}

int ppocr_from_sxml( CPPOCR * cppocr, char const * xmlfile ) noexcept
{
    return -5;
}

void ppocr_destroy( CPPOCR cppocr ) noexcept
{
  if ( !cppocr )
    return;

  delete cppocr->exe;
  delete cppocr;
}

void ppocr_print_result( PPPOcrResult results ) noexcept
{
  int i = 0;
  PPPOcrResult current = results;
  while ( current )
  {
    printf( "%d\t", i++ );

    // rec
    if (current->score != -1.0) {
      printf( "rec text: %s rec score: %f ",
               current->text, current->score );
    }

    // cls
    if (current->cls_label != -1) {
      printf( "cls label: %d rec score: %f",
              current->cls_label, current->cls_score );
    }

    printf( "\n" );

    current = current->next;
  }
}

int ocr(std::vector<cv::String> &cv_all_img_names, CPPOCR cppocr,
        std::vector<std::vector<OCRPredictResult>> &ocr_results) noexcept {
  PPOCR &ocr = *(cppocr->exe);
  Args const & args = ocr.args();

  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data)
      continue;
    img_list.emplace_back(img);
    img_names.emplace_back(cv_all_img_names[i]);
  }

  if ( img_list.empty() )
    return -8;

  ocr_results = ocr.ocr(img_list);

  for (int i = 0; i < img_names.size(); ++i) {
    if (args.visualize && args.det) {
      std::string file_name = Utility::basename(img_names[i]);
      cv::Mat srcimg = img_list[i];
      Utility::VisualizeBboxes(srcimg, ocr_results[i],
                               args.output + "/" + file_name);
    }
  }

  return 0;
}

int structure(std::vector<cv::String> &cv_all_img_names, CPPOCR cppocr,
              std::vector<std::vector<OCRPredictResult>> &ocr_results) noexcept {
  PaddleStructure &engine = *dynamic_cast<PaddleStructure*>(cppocr->exe);
  Args const & args = engine.args();

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    // std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      // std::cerr << "[ERROR] image read failed! image path: "
      //           << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine.structure(img);

    for (size_t j = 0; j < structure_results.size(); ++j) {
      if (structure_results[j].type == "table") {
        // std::cout << structure_results[j].html << std::endl;
        if (structure_results[j].cell_box.size() > 0 && args.visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   args.output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
      }
      ocr_results.emplace_back(std::move(structure_results[j].text_res));
    }
  }

  return 0;
}

int ppocr_cmd( CPPOCR cppocr, PPPOcrResult * result ) noexcept {
  return ppocr_exe( cppocr, cppocr->exe->args().image_dir.c_str(), result );
}

int ppocr_exe( CPPOCR cppocr, char const * image_dir, PPPOcrResult * result ) noexcept {

  if (!Utility::PathExists(image_dir)) {
    return -3;
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(image_dir, cv_all_img_names);

  Args const & args = cppocr->exe->args();

  if (!Utility::PathExists(args.output))
    Utility::CreateDir(args.output);

  int ret = 7;
  std::vector<std::vector<OCRPredictResult>> ocr_results;

  if (args.type == 1)
    ret = ocr(cv_all_img_names, cppocr, ocr_results);
  else if (args.type == 2)
    ret = structure(cv_all_img_names, cppocr, ocr_results);

  if ( ret || !result )
    return ret;

  *result = nullptr;

  PPPOcrResult * recent = result;

  for ( int i = 0; i < ocr_results.size(); ++i )
  {
    std::vector<OCRPredictResult> const & results = ocr_results[i];
    for ( int j = 0; j < results.size(); ++j )
    {
      if ( results[j].score < 0.1 )
        continue;

      if ( *recent == nullptr )
      {
        *recent = new PPOcrResult;
        (*recent)->next = nullptr;
      }

      (*recent)->text = new char[results[j].text.size() + 1];
      strncpy( (*recent)->text, results[j].text.c_str(), results[j].text.size()+1 );
      (*recent)->score = results[j].score;
      (*recent)->cls_score = results[j].cls_score;
      (*recent)->cls_label = results[j].cls_label;
      recent = &(*recent)->next;
    }
  }

  return 0;
}

void ppocr_free( PPPOcrResult result ) noexcept
{
  PPPOcrResult recent = result;
  while(recent)
  {
    PPPOcrResult toDel = recent;
    recent = recent->next;
    delete [] toDel->text;
    delete toDel;
  }
}
