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
#include "ppocr_c.h"
#include <paddlestructure.h>
#include "opencv2/imgcodecs.hpp"
#include <args.h>

using namespace PaddleOCR;

int check_params( Args const & args ) {
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


struct PPOCRC {
  Args   args;
  PPOCR* exe;
};

int ppocr_from_Args( CPPOCR * cppocr, Args const & args )
{
  int ret = check_params(args);
  if ( ret )
    return ret;

  PPOCR * ocr;

  if (args.type == "ocr")
    ocr = new PaddleOCR::PPOCR(args);
  else if (args.type == "structure")
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

  (*cppocr)->args = args;
  (*cppocr)->exe = ocr;

  return 0;
}


int ppocr_from_args( CPPOCR * cppocr, int argc, char ** argv )
{
  return ppocr_from_Args( cppocr, Args( argc, argv ) );
}

int ppocr_from_inis( CPPOCR * cppocr, const char *inis )
{
  return ppocr_from_Args( cppocr, Args(inis) );
}

int ppocr_from_sxml( CPPOCR * cppocr, char const * xmlfile )
{
    return -5;
}

void ppocr_destroy( CPPOCR cppocr )
{
  if ( !cppocr )
    return;

  delete cppocr->exe;
  delete cppocr;
}

void ppocr_print_result( PPPOcrResult results )
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
        std::vector<std::vector<OCRPredictResult>> &ocr_results) {
  Args const & args = cppocr->args;
  PPOCR &ocr = *(cppocr->exe);

  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      // std::cerr << "[ERROR] image read failed! image path: "
      //           << cv_all_img_names[i] << std::endl;
      continue;
    }
    img_list.emplace_back(img);
    img_names.emplace_back(cv_all_img_names[i]);
  }

  if ( img_list.empty() )
    return -8;

  ocr_results = ocr.ocr(img_list, args.det, args.rec, args.cls);

  for (int i = 0; i < img_names.size(); ++i) {
    // std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    // Utility::print_result(ocr_results[i]);
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
              std::vector<std::vector<OCRPredictResult>> &ocr_results) {
  PaddleStructure * _ = dynamic_cast<PaddleStructure*>(cppocr->exe);
  if ( !_ )
    return -3;

  Args const & args = cppocr->args;
  PaddleStructure &engine = *_;

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    // std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      // std::cerr << "[ERROR] image read failed! image path: "
      //           << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, args.layout, args.table, args.det && args.rec);

    for (int j = 0; j < structure_results.size(); j++) {
      // std::cout << j << "\ttype: " << structure_results[j].type
      //           << ", region: [";
      // std::cout << structure_results[j].box[0] << ","
      //           << structure_results[j].box[1] << ","
      //           << structure_results[j].box[2] << ","
      //           << structure_results[j].box[3] << "], score: ";
      // std::cout << structure_results[j].confidence << ", res: ";

      if (structure_results[j].type == "table") {
        // std::cout << structure_results[j].html << std::endl;
        if (structure_results[j].cell_box.size() > 0 && args.visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   args.output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
      }  {
        // std::cout << "count of ocr result is : "
        //           << structure_results[j].text_res.size() << std::endl;
        // if (structure_results[j].text_res.size() > 0) {
        //   std::cout << "********** print ocr result "
        //             << "**********" << std::endl;
        //   Utility::print_result(structure_results[j].text_res);
        //   std::cout << "********** end print ocr result "
        //             << "**********" << std::endl;
        // }
      }
      ocr_results.emplace_back(std::move(structure_results[j].text_res));
    }
  }

  return 0;
}

int ppocr_exe( CPPOCR cppocr, char const * image_dir,
               PPPOcrResult * result ) {
  Args & args = cppocr->args;
  args.image_dir = image_dir;
  return ppocr_cmd( cppocr, result );
}

int ppocr_cmd( CPPOCR cppocr, PPPOcrResult * result ) {
  Args const & args = cppocr->args;

  if (!Utility::PathExists(args.image_dir)) {
    // std::cerr << "[ERROR] image path not exist! image_dir: " << args.image_dir
    //           << std::endl;
    return -3;
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(args.image_dir, cv_all_img_names);

  // std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  if (!Utility::PathExists(args.output))
    Utility::CreateDir(args.output);

  int ret = 7;
  std::vector<std::vector<OCRPredictResult>> ocr_results;

  if (args.type == "ocr")
    ret = ocr(cv_all_img_names, cppocr, ocr_results);
  else if (args.type == "structure")
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

void ppocr_free( PPPOcrResult result )
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
