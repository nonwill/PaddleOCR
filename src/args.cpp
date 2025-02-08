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

#include <args.h>

#include "getopt.hh"
#include <iostream>
#include <string.h>

namespace PaddleOCR {

Args::Args() noexcept {
#define DEFINE_bool(x, v, d) x = v;
#define DEFINE_int32(x, v, d) x = v;
#define DEFINE_double(x, v, d) x = v;
#define DEFINE_string(x, v, d) x = v;
#define DEFINE_void DEFINE_bool
#include "args_pri.h"
#undef DEFINE_void
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
}

void ArgsHelp(std::ostream &out) noexcept {
#define DEFINE_void(x, v, d)                                                   \
  out << "  --" << #x << ": " << d << std::endl;

#define DEFINE_bool(x, v, d)                                                   \
  out << "  --" << #x << ": " << d << std::endl                                \
      << "    type: bool  default: " << (v ? "true" : "false") << std::endl;

#define DEFINE_int32(x, v, d)                                                  \
  out << "  --" << #x << ": " << d << std::endl                                \
      << "    type: int32  default: " << v << std::endl;

#define DEFINE_double(x, v, d)                                                 \
  out << "  --" << #x << ": " << d << std::endl                                \
      << "    type: double  default: " << v << std::endl;

#define DEFINE_string(x, v, d)                                                 \
  out << "  --" << #x << ": " << d << std::endl                                \
      << "    type: string  default: \"" << v << "\"" << std::endl;

  out << "Flags from PaddleOCR::Args:" << std::endl;
#include "args_pri.h"
  out << std::endl;

#undef DEFINE_void
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
}

int Args::parseArgv(int argc, char **argv) noexcept {
#define DEFINE_bool(x, v, d) OPTION_FOR_CONTEXT_BOOL(x),
#define DEFINE_int32(x, v, d) OPTION_FOR_CONTEXT_SINT(x),
#define DEFINE_double(x, v, d) OPTION_FOR_CONTEXT_DOUBLE(x),
#define DEFINE_string(x, v, d) OPTION_FOR_CONTEXT_STRING(x),
#define DEFINE_void DEFINE_bool
  OptPlus::option const long_options[] = {{"help", OptPlus::no, 0, 'h', 0},
#include "args_pri.h"
                                        {0, OptPlus::no, 0, 0, 0}};
#undef DEFINE_void
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string

  OptPlus optpp;

  for (;;) {
    int option_index = 0;
    int ret = optpp.travel_long(argc, argv, "h", long_options, &option_index);
    if (ret < 0)
      break;

    OptPlus::option const &opt = long_options[option_index];
    if (!opt.name)
      continue;

    if (!opt.context) {
      if (ret == 'h')
        help = true;
      else if (option_index)
        fprintf(stderr, "OptPlus null context: %s val=%d\n", opt.name, opt.val);
    } else if (opt.val == OptPlus::v_bool) {
      *((bool *)opt.context) = optpp.as_bool();
    } else if (opt.val == OptPlus::v_sint) {
      *((int *)opt.context) = optpp.as_sint();
    // } else if (opt.val == OptPlus::v_float) {
    //   *((float *)opt.context) = optpp.arg_as_double();
    } else if (opt.val == OptPlus::v_double) {
      *((double *)opt.context) = optpp.as_double();
    } else if (opt.val == OptPlus::v_string) {
      *((std::string *)opt.context) = optpp.as_str();
    } else {
      fprintf(stderr, "OptPlus ignored context: %s type=%d\n", opt.name, opt.val);
    }
  }

  int optind = optpp.ind();
  if (optind < argc) {
    fprintf(stderr, "non-option ARGV-elements: ");
    while (optind < argc)
      fprintf(stderr, "%s ", argv[optind++]);
    fprintf(stderr, "\n");
  }

  return 0;
}

int Args::parseInis(char const *inis) noexcept {
  char const *lb = inis;
  char const *le = inis;

  for (;;) {
    le = strstr(lb, "\n");
    if (le == nullptr)
      break;

    std::string cats(lb, le - lb);
    lb = le + 1;

    size_t pos = cats.find('=');
    if (pos == std::string::npos)
      continue;

    std::string name = cats.substr(0, pos);
    std::string value = cats.substr(pos + 1);
    if (value.empty())
      continue;

#define DEFINE_bool(x, v, d)                                                   \
  if (name == #x) {                                                            \
    x = strcmp(value.c_str(), "true") == 0 || atoi(value.c_str()) != 0;        \
    continue;                                                                  \
  }
#define DEFINE_int32(x, v, d)                                                  \
  if (name == #x) {                                                            \
    x = atoi(value.c_str());                                                   \
    continue;                                                                  \
  }
#define DEFINE_double(x, v, d)                                                 \
  if (name == #x) {                                                            \
    x = atof(value.c_str());                                                   \
    continue;                                                                  \
  }
#define DEFINE_string(x, v, d)                                                 \
  if (name == #x) {                                                            \
    x = value;                                                                 \
    continue;                                                                  \
  }
#define DEFINE_void DEFINE_bool
#include <args_pri.h>
#undef DEFINE_void
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
  }

  return 0;
}

} // namespace PaddleOCR
