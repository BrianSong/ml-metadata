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
#include "ml_metadata/tools/mlmd_bench/read_events_workload.h"

#include <random>
#include <vector>

#include "absl/time/clock.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

template <typename T>
void InitializeReadRequest(ReadEventsWorkItemType& read_request) {
  read_request.emplace<T>();
}

}  // namespace

ReadEvents::ReadEvents(const ReadEventsConfig& read_events_config,
                       const int64 num_operations)
    : read_events_config_(read_events_config), num_operations_(num_operations) {
  switch (read_events_config.specification()) {
    case ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS: {
      name_ = "read_events_by_artifact_ids";
      break;
    }
    case ReadEventsConfig::EVENTS_BY_EXECUTION_IDS: {
      name_ = "read_events_by_execution_ids";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for ReadEvents!";
  }
}

tensorflow::Status ReadEvents::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<NodeType> existing_nodes;
  TF_RETURN_IF_ERROR(GetExistingNodes(read_events_config_.specification() % 3,
                                      store, existing_nodes));

  std::uniform_int_distribution<int64> uniform_dist_node_index{
      0, (int64)(existing_nodes.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    ReadEventsWorkItemType read_request;
    const int64 node_index = uniform_dist_node_index(gen);
    switch (read_events_config_.specification()) {
      case ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS: {
        InitializeReadRequest<GetEventsByArtifactIDsRequest>(read_request);
        Artifact picked_node = absl::get<Artifact>(existing_nodes[node_index]);
        absl::get<GetEventsByArtifactIDsRequest>(read_request)
            .add_artifact_ids(picked_node.id());
        break;
      }
      case ReadEventsConfig::EVENTS_BY_EXECUTION_IDS: {
        InitializeReadRequest<GetEventsByExecutionIDsRequest>(read_request);
        Execution picked_node =
            absl::get<Execution>(existing_nodes[node_index]);
        absl::get<GetEventsByExecutionIDsRequest>(read_request)
            .add_execution_ids(picked_node.id());
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for ReadEvents!";
    }
    work_items_.emplace_back(read_request, curr_bytes);
  }

  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadEvents::RunOpImpl(const int64 work_items_index,
                                         MetadataStore* store) {
  switch (read_events_config_.specification()) {
    case ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS: {
      GetEventsByArtifactIDsRequest request =
          absl::get<GetEventsByArtifactIDsRequest>(
              work_items_[work_items_index].first);
      GetEventsByArtifactIDsResponse response;
      return store->GetEventsByArtifactIDs(request, &response);
      break;
    }
    case ReadEventsConfig::EVENTS_BY_EXECUTION_IDS: {
      GetEventsByExecutionIDsRequest request =
          absl::get<GetEventsByExecutionIDsRequest>(
              work_items_[work_items_index].first);
      GetEventsByExecutionIDsResponse response;
      return store->GetEventsByExecutionIDs(request, &response);
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status ReadEvents::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string ReadEvents::GetName() { return name_; }

}  // namespace ml_metadata
