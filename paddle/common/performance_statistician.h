// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(PADDLE_WITH_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "paddle/common/enforce.h"

namespace common {

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
using TimeDuration = std::chrono::duration<double, std::micro>;

struct TimePointInfo {
  bool is_start;
  TimePoint time_point;
};

class PerformanceStatistician {
public:
  using TimeRecordPerThread =
      std::unordered_map<std::thread::id, std::vector<TimePointInfo>>;

  static PerformanceStatistician &Instance() {
    static PerformanceStatistician instance;
    return instance;
  }

  void InsertTimePoint(const std::string &label, bool is_start) {
    std::thread::id thread_id = std::this_thread::get_id();
    if (is_start) {
      std::lock_guard<std::mutex> lck_guard(record_mtx_);
      TimePoint time_point = std::chrono::steady_clock::now();
      record_[label][thread_id].push_back(TimePointInfo{is_start, time_point});
    } else {
      TimePoint time_point = std::chrono::steady_clock::now();
      std::lock_guard<std::mutex> lck_guard(record_mtx_);
      record_[label][thread_id].push_back(TimePointInfo{is_start, time_point});
    }
  }

  void Start(const std::string &label) {
    InsertTimePoint(label, /* is_start = */ true);
  }

  void End(const std::string &label) {
    InsertTimePoint(label, /* is_start = */ false);
  }

  void CudaStart(const std::string &label) {
#if defined(PADDLE_WITH_CUDA)
    std::lock_guard<std::mutex> lck_guard(record_mtx_);
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cuda_events_[label].first = e_start;
    cuda_events_[label].second = e_stop;
    cudaEventRecord(cuda_events_[label].first, 0);
#endif
  }

  void CudaEnd(const std::string &label) {
#if defined(PADDLE_WITH_CUDA)
    std::lock_guard<std::mutex> lck_guard(record_mtx_);
    PADDLE_ENFORCE_NE(cuda_events_.count(label), 0,
                      common::errors::InvalidArgument(
                          "Key for cuda time record does not exist"));

    cudaEventRecord(cuda_events_[label].second, 0);
    cudaEventSynchronize(cuda_events_[label].second);
    float event_duration;
    cudaEventElapsedTime(&event_duration, cuda_events_[label].first,
                         cuda_events_[label].second);
    std::thread::id thread_id = std::this_thread::get_id();
    TimePoint start_time_point = std::chrono::steady_clock::now();
    TimePoint end_time_point =
        start_time_point + std::chrono::nanoseconds(
                               static_cast<int64_t>(event_duration * 1000000));
    record_[label][thread_id].push_back(TimePointInfo{true, start_time_point});
    record_[label][thread_id].push_back(TimePointInfo{false, end_time_point});
    cudaEventDestroy(cuda_events_[label].first);
    cudaEventDestroy(cuda_events_[label].second);
    cuda_events_.erase(label);
#endif
  }

  std::vector<TimePointInfo> Record(const std::string &label) const {
    if (record_.count(label) == 0)
      return {};
    std::vector<TimePointInfo> record_all_threads;
    for (const auto &time_points : record_.at(label)) {
      record_all_threads.insert(record_all_threads.end(),
                                time_points.second.begin(),
                                time_points.second.end());
    }
    return record_all_threads;
  }

  std::vector<TimePointInfo>
  RecordWithSubLabel(const std::string &sub_label) const {
    const auto IsSubStr = [](const std::string &sub, const std::string &str) {
      return str.find(sub) != std::string::npos;
    };

    std::vector<TimePointInfo> records;
    std::vector<std::string> labels = Labels();
    for (const auto &label : labels) {
      if (IsSubStr(sub_label, label)) {
        auto appends = Record(label);
        records.insert(records.end(), appends.begin(), appends.end());
      }
    }
    return records;
  }

  std::vector<std::string> Labels() const {
    std::vector<std::string> labels;
    for (const auto &item : record_) {
      labels.push_back(item.first);
    }
    return labels;
  }

  void Reset(const std::string &label) {
    std::lock_guard<std::mutex> lck_guard(record_mtx_);
    record_[label].clear();
  }

  void Reset() {
    std::lock_guard<std::mutex> lck_guard(record_mtx_);
    record_.clear();
  }

  void SetGraphNodesNum(int graph_nodes_num) {
    graph_nodes_num_ = graph_nodes_num;
  }

  int GetGraphNodesNum() const { return graph_nodes_num_; }

private:
  PerformanceStatistician() = default;
  ~PerformanceStatistician() = default;
  PerformanceStatistician(const PerformanceStatistician &) = delete;
  void operator=(const PerformanceStatistician &) = delete;

private:
  std::unordered_map<std::string, TimeRecordPerThread> record_;
#if defined(PADDLE_WITH_CUDA)
  std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>>
      cuda_events_;
#endif
  std::mutex record_mtx_;
  int graph_nodes_num_ = 25;
};

class PerformanceReporter {
public:
  static std::vector<TimeDuration>
  ExtractDuration(const std::vector<TimePointInfo> &records,
                  bool contain_recursive = false);

  static TimeDuration Sum(const std::vector<TimeDuration> &records);

  static TimeDuration Mean(const std::vector<TimeDuration> &records);

  static TimeDuration TrimMean(const std::vector<TimeDuration> &records);

  static TimeDuration Max(const std::vector<TimeDuration> &records);

  static TimeDuration Min(const std::vector<TimeDuration> &records);

  static std::vector<TimeDuration>
  TopK(const std::vector<TimeDuration> &records, int top_count);

  static std::string Report(const std::vector<TimePointInfo> &records);

  static std::string Report(const PerformanceStatistician &stat);

  static void WriteToFile(const std::string &file, const std::string &report);
};

void PerformanceStatisticsStart(const std::string &label);

void PerformanceStatisticsEnd(const std::string &label);

} // namespace common
