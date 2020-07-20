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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_FILL_EVENTS_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_FILL_EVENTS_WORKLOAD_H

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

class FillEvents : public Workload<PutEventsRequest> {
 public:
  FillEvents(const FillEventsConfig& fill_events_config, int64 num_operations);
  ~FillEvents() override = default;

 protected:
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  tensorflow::Status RunOpImpl(int64 i, MetadataStore* store) final;

  tensorflow::Status TearDownImpl() final;

  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const FillEventsConfig fill_events_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  std::string name_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_FILL_EVENTS_WORKLOAD_H
