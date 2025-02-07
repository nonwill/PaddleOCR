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

#include "getopt.h"
#include <iostream>
#include <string.h>

namespace PaddleOCR {

Args::Args() noexcept
{
#define DEFINE_bool(x,v,d) x = v;
#define DEFINE_int32(x,v,d) x = v;
#define DEFINE_double(x,v,d) x = v;
#define DEFINE_string(x,v,d) x = v;
#include "args_pri.h"
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
}

void ArgsHelp(std::ostream & out) noexcept
{
#define DEFINE_bool(x,v,d) out << "  --"  \
        << #x << ": " << d << std::endl \
        << "    type: bool  default: " \
        << (v ? "true" : "false") << std::endl;

#define DEFINE_int32(x,v,d) out << "  --" \
        << #x << ": " << d << std::endl \
        << "    type: int32  default: " \
        << v << std::endl;

#define DEFINE_double(x,v,d) out << "  --" \
        << #x << ": " << d << std::endl \
        << "    type: double  default: " \
        << v << std::endl;

#define DEFINE_string(x,v,d) out << "  --" \
        << #x << ": " << d << std::endl \
        << "    type: string  default: \"" \
        << v << "\"" << std::endl;

  out << "Flags from PaddleOCR::Args:" << std::endl;
#include "args_pri.h"
  out << std::endl;

#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
}

int Args::parseArgv( int argc, char** argv ) noexcept
{
  struct option const long_options[] =
  {
    { "help", 0, 0, 'h', 0 },
#define DEFINE_bool(x,v,d) {#x, 2, 0, 1, &x },
#define DEFINE_int32(x,v,d) {#x, 2, 0, 2, &x },
#define DEFINE_double(x,v,d) {#x, 2, 0, 3, &x },
#define DEFINE_string(x,v,d) {#x, 2, 0, 4, &x },
#include "args_pri.h"
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
    { 0, 0, 0, 0, 0 }
  };

  int ret;
  int option_index;

  for(;;)
  {
    option_index = 0;
    ret = getopt_long(argc, argv, "h", long_options, &option_index);
    if ( ret < 0 )
      break;

    struct option const opt = long_options[option_index];
    if ( !opt.name )
      continue;

    if ( !opt.context )
    {
      if ( ret == 'h' )
        ArgsHelp(std::cout);
      else
      if ( option_index )
        printf("non-context option: %s val=%d\n", opt.name, opt.val);
    }
    else
    if ( opt.val == 1 )
    {
      *((bool*)opt.context) = strcmp( optarg, "true" ) == 0 || atoi( optarg ) != 0;
    }
    else
    if ( opt.val == 2 )
    {
      *((int*)opt.context) = atoi( optarg );
    }
    else
    if ( opt.val == 3 )
    {
      *((double*)opt.context) = atof( optarg );
    }
    else
    if ( opt.val == 4 )
    {
      *((std::string*)opt.context) = optarg;
    }
    else
    {
      printf("notype-bind option: %s type=%d\n", opt.name, opt.val);
    }
  }

  if (optind < argc)
  {
    printf("non-option ARGV-elements: ");
    while (optind < argc)
      printf("%s ", argv[optind++]);
    printf("\n");
  }

  return 0;
}

int Args::parseInis( char const * inis ) noexcept
{
  char const * lb = inis;
  char const * le = inis;

  for(;;)
  {
    le = strstr( lb, "\n" );
    if ( le == nullptr )
      break;

    std::string cats( lb, le - lb );
    lb = le + 1;

    size_t pos = cats.find( '=' );
    if ( pos == std::string::npos )
      continue;

    std::string name = cats.substr(0, pos);
    std::string value = cats.substr(pos+1);
    if ( value.empty() )
      continue;

#define DEFINE_bool(x,v,d) \
    if ( name == #x ) { \
      x = strcmp( value.c_str(), "true" ) == 0 \
                     || atoi( value.c_str() ) != 0; \
      continue; \
    }
#define DEFINE_int32(x,v,d) \
    if ( name == #x ) { \
      x = atoi( value.c_str() ); \
      continue; \
    }
#define DEFINE_double(x,v,d) \
    if ( name == #x ) { \
      x = atof( value.c_str() ); \
      continue; \
    }
#define DEFINE_string(x,v,d) \
    if ( name == #x ) { \
      x = value; \
      continue; \
    }
#include <args_pri.h>
#undef DEFINE_bool
#undef DEFINE_int32
#undef DEFINE_double
#undef DEFINE_string
  }
  return 0;
}

}
