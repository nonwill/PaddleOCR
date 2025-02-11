// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "paddle/common/flags.h"
#include "paddle/utils/string/printf.h"

namespace paddle {

namespace string {

inline std::string black() { return "\e[30m"; }
inline std::string red() { return "\e[31m"; }
inline std::string b_red() { return "\e[41m"; }
inline std::string green() { return "\e[32m"; }
inline std::string yellow() { return "\e[33m"; }
inline std::string blue() { return "\e[34m"; }
inline std::string purple() { return "\e[35m"; }
inline std::string cyan() { return "\e[36m"; }
inline std::string light_gray() { return "\e[37m"; }
inline std::string white() { return "\e[37m"; }
inline std::string light_red() { return "\e[91m"; }
inline std::string dim() { return "\e[2m"; }
inline std::string bold() { return "\e[1m"; }
inline std::string underline() { return "\e[4m"; }
inline std::string blink() { return "\e[5m"; }
inline std::string reset() { return "\e[0m"; }

using TextBlock = std::pair<std::string, std::string>;

struct Style {
  static std::string info() { return black(); }
  static std::string warn() { return b_red(); }
  static std::string suc() { return green(); }
  static std::string H1() { return bold() + purple(); }
  static std::string H2() { return green(); }
  static std::string H3() { return green(); }
  static std::string detail() { return light_gray(); }
};

template <typename... Args>
static void PrettyLogEndl(const std::string &style, const char *fmt,
                          const Args &...args) {
  std::cerr << style << Sprintf(fmt, args...) << reset() << std::endl;
}
template <typename... Args>
static void PrettyLog(const std::string &style, const char *fmt,
                      const Args &...args) {
  std::cerr << style << Sprintf(fmt, args...) << reset();
}

template <typename... Args>
static void PrettyLogInfo(const char *fmt, const Args &...args) {
  PrettyLogEndl(Style::info(), fmt, args...);
}
template <typename... Args>
static void PrettyLogDetail(const char *fmt, const Args &...args) {
  PrettyLogEndl(Style::detail(), fmt, args...);
}
template <typename... Args>
static void PrettyLogH1(const char *fmt, const Args &...args) {
  PrettyLogEndl(Style::H1(), fmt, args...);
}
template <typename... Args>
static void PrettyLogH2(const char *fmt, const Args &...args) {
  PrettyLogEndl(Style::H2(), fmt, args...);
}

} // namespace string
} // namespace paddle
