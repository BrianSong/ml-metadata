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
#include <fcntl.h>

#include <iostream>
#include <random>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "ml_metadata/tools/mlmd_bench/benchmark.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/thread_runner.h"

namespace ml_metadata {
namespace {

// Initializes the mlmd_bench_config with the specified config pbtxt file.
void InitMLMDBenchConfigFromPbtxtFile(char** argv,
                                      MLMDBenchConfig& mlmd_bench_config) {
  int file_descriptor = open(argv[1], O_RDONLY);
  if (file_descriptor < 0) {
    LOG(ERROR) << "Cannot open the config. file: " << argv[1] << " !";
  }
  google::protobuf::io::FileInputStream file_input(file_descriptor);
  if (!google::protobuf::TextFormat::Parse(&file_input, &mlmd_bench_config)) {
    LOG(ERROR) << "Fail to parse the config. file: " << argv[1] << " !";
  }
}

}  // namespace
}  // namespace ml_metadata

int main(int argc, char** argv) {
  ml_metadata::MLMDBenchConfig mlmd_bench_config;
  ml_metadata::InitMLMDBenchConfigFromPbtxtFile(argv, mlmd_bench_config);
  srand(time(NULL));
  // Feeds the config. into the benchmark for generating executable workloads.
  ml_metadata::Benchmark benchmark(mlmd_bench_config);
  ml_metadata::ThreadRunner runner(mlmd_bench_config);
  // Executes the workloads inside the benchmark with the thread runner.
  runner.Run(benchmark);

  return 0;
}
