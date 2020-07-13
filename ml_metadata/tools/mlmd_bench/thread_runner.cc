/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "ml_metadata/tools/mlmd_bench/thread_runner.h"

#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/benchmark.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/stats.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace ml_metadata {

ThreadRunner::ThreadRunner(const MLMDBenchConfig& mlmd_bench_config)
    : num_threads_(mlmd_bench_config.thread_env_config().num_threads()),
      mlmd_config_(mlmd_bench_config.mlmd_config()) {}

// Execution unit of mlmd_bench.
tensorflow::Status ThreadRunner::Run(Benchmark& benchmark) {
  // Loops over all the workloads inside the benchmark and executes them one by
  // one.
  for (auto& workload : benchmark.workloads()) {
    Stats thread_stats_list[num_threads_];
    int64 op_per_thread = workload.second / num_threads_;
    std::unique_ptr<MetadataStore> set_up_store;
    TF_RETURN_IF_ERROR(CreateMetadataStore(mlmd_config_, &set_up_store));
    TF_RETURN_IF_ERROR(workload.first->SetUp(set_up_store.get()));
    {
      tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(),
                                          "mlmd_bench", num_threads_);
      int64 total_done = 0;
      int64 thread = 0;
      for (int64 i = 0; i < num_threads_; ++i) {
        pool.Schedule([&]() {
          // Each thread uses a different MLMD client instance to talk to the
          // same back-end
          lock_.Lock();
          int64 curr_thread = thread;
          thread++;
          lock_.Unlock();
          std::unique_ptr<MetadataStore> store;
          CreateMetadataStore(mlmd_config_, &store);
          int64 start_index = op_per_thread * curr_thread;
          thread_stats_list[curr_thread].Start();
          // Executes the current workload by the specified index of work item.
          int64 work_items_index = start_index;
          while (work_items_index < start_index + op_per_thread) {
            // Each operation will has a op_stats.
            OpStats op_stats;
            tensorflow::Status status =
                workload.first->RunOp(work_items_index, store.get(), op_stats);
            if (!status.ok()) {
              continue;
            }
            work_items_index++;
            lock_.Lock();
            total_done++;
            lock_.Unlock();
            // Updates the thread stats using the op_stats.
            thread_stats_list[curr_thread].Update(op_stats, total_done);
          }
          thread_stats_list[curr_thread].Stop();
        });
      }
    }
    TF_RETURN_IF_ERROR(workload.first->TearDown());
    // Merges all the thread stats of the current workload.
    for (int64 i = 1; i < num_threads_; ++i) {
      thread_stats_list[0].Merge(thread_stats_list[i]);
    }
    // Reports the metrics of interests.
    thread_stats_list[0].Report(workload.first->GetName());
  }
  return tensorflow::Status::OK();
}

}  // namespace ml_metadata
