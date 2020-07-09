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
#include "ml_metadata/tools/mlmd_bench/benchmark.h"

#include <gtest/gtest.h>
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"

namespace ml_metadata {
namespace {

// Tests the CreateWorkload() of Benchmark class.
TEST(BenchmarkTest, CreatWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config;
  WorkloadConfig* workload_config = mlmd_bench_config.add_workload_configs();
  FillTypesConfig* fill_types_config =
      workload_config->mutable_fill_types_config();
  fill_types_config->set_specification(FillTypesConfig::ARTIFACT_TYPE);

  Benchmark benchmark(mlmd_bench_config);
  // Checks that indeed one workload has been created according to the input
  // workload config.
  EXPECT_STREQ(benchmark.workloads()[0].first->GetName().c_str(),
               "fill_artifact_type");
}

}  // namespace
}  // namespace ml_metadata
