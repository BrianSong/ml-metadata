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
#include "ml_metadata/tools/mlmd_bench/fill_events_workload.h"

#include <random>
#include <vector>

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

tensorflow::Status GenerateEvent(const FillEventsConfig& fill_events_config,
                                 const int64 artifact_node_id,
                                 const int64 execution_node_id,
                                 std::unordered_set<int64>& output_artifact_ids,
                                 Event* event, int64& curr_bytes) {
  switch (fill_events_config.specification()) {
    case FillEventsConfig::INPUT: {
      event->set_type(Event::INPUT);
      break;
    }
    case FillEventsConfig::OUTPUT: {
      event->set_type(Event::OUTPUT);
      if (output_artifact_ids.find(artifact_node_id) !=
          output_artifact_ids.end()) {
        return tensorflow::errors::AlreadyExists(
            ("Current artifact has been outputted by another execution "
             "already!"));
      }
      output_artifact_ids.insert(artifact_node_id);
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillEvents!";
  }
  event->set_artifact_id(artifact_node_id);
  event->set_execution_id(execution_node_id);
  curr_bytes += 8 * 2;
  return tensorflow::Status::OK();
}

}  // namespace

FillEvents::FillEvents(const FillEventsConfig& fill_events_config,
                       int64 num_operations)
    : fill_events_config_(fill_events_config), num_operations_(num_operations) {
  switch (fill_events_config_.specification()) {
    case FillEventsConfig::INPUT: {
      name_ = "fill_input_event";
      break;
    }
    case FillEventsConfig::OUTPUT: {
      name_ = "fill_output_event";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillEvents!";
  }
}

tensorflow::Status FillEvents::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<NodeType> existing_artifact_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(/*specification=*/0, store, existing_artifact_nodes));
  std::vector<NodeType> existing_execution_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(/*specification=*/1, store, existing_execution_nodes));

  std::uniform_int_distribution<int64> uniform_dist_artifact_index{
      0, (int64)(existing_artifact_nodes.size() - 1)};
  std::uniform_int_distribution<int64> uniform_dist_execution_index{
      0, (int64)(existing_execution_nodes.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  int64 i = 0;
  while (i < num_operations_) {
    curr_bytes = 0;
    PutEventsRequest put_request;
    const int64 artifact_index = uniform_dist_artifact_index(gen);
    const int64 execution_index = uniform_dist_execution_index(gen);
    if (!GenerateEvent(
             fill_events_config_,
             absl::get<Artifact>(existing_artifact_nodes[artifact_index]).id(),
             absl::get<Execution>(existing_execution_nodes[execution_index])
                 .id(),
             output_artifact_ids_, put_request.add_events(), curr_bytes)
             .ok()) {
      continue;
    }
    work_items_.emplace_back(put_request, curr_bytes);
    i++;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status FillEvents::RunOpImpl(const int64 work_items_index,
                                         MetadataStore* store) {
  PutEventsRequest put_request = work_items_[work_items_index].first;
  PutEventsResponse put_response;
  TF_RETURN_IF_ERROR(store->PutEvents(put_request, &put_response));
  return tensorflow::Status::OK();
}

tensorflow::Status FillEvents::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillEvents::GetName() { return name_; }

}  // namespace ml_metadata
