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
#include "ml_metadata/tools/mlmd_bench/read_nodes_via_context_edges_workload.h"

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
void InitializeReadRequest(ReadNodesViaContextEdgesWorkItemType& read_request) {
  read_request.emplace<T>();
}

}  // namespace

ReadNodesViaContextEdges::ReadNodesViaContextEdges(
    const ReadNodesViaContextEdgesConfig& read_nodes_via_context_edges_config,
    const int64 num_operations)
    : read_nodes_via_context_edges_config_(read_nodes_via_context_edges_config),
      num_operations_(num_operations) {
  switch (read_nodes_via_context_edges_config_.specification()) {
    case ReadNodesViaContextEdgesConfig::ARTIFACTS_BY_CONTEXT: {
      name_ = "read_artifacts_by_context";
      break;
    }
    case ReadNodesViaContextEdgesConfig::EXECUTIONS_BY_CONTEXT: {
      name_ = "read_executions_by_context";
      break;
    }
    case ReadNodesViaContextEdgesConfig::CONTEXTS_BY_ARTIFACT: {
      name_ = "read_contexts_by_artifact";
      break;
    }
    case ReadNodesViaContextEdgesConfig::CONTEXTS_BY_EXECUTION: {
      name_ = "read_contexts_by_execution";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for ReadNodesViaContextEdges!";
  }
}

tensorflow::Status ReadNodesViaContextEdges::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<NodeType> existing_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(read_nodes_via_context_edges_config_.specification() % 3,
                       store, existing_nodes));

  std::uniform_int_distribution<int64> uniform_dist_node_index{
      0, (int64)(existing_nodes.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    ReadNodesViaContextEdgesWorkItemType read_request;
    const int64 node_index = uniform_dist_node_index(gen);
    switch (read_nodes_via_context_edges_config_.specification()) {
      case ReadNodesViaContextEdgesConfig::ARTIFACTS_BY_CONTEXT: {
        InitializeReadRequest<GetArtifactsByContextRequest>(read_request);
        Context picked_node = absl::get<Context>(existing_nodes[node_index]);
        absl::get<GetArtifactsByContextRequest>(read_request)
            .set_context_id(picked_node.id());
        break;
      }
      case ReadNodesViaContextEdgesConfig::EXECUTIONS_BY_CONTEXT: {
        InitializeReadRequest<GetExecutionsByContextRequest>(read_request);
        Context picked_node = absl::get<Context>(existing_nodes[node_index]);
        absl::get<GetExecutionsByContextRequest>(read_request)
            .set_context_id(picked_node.id());
        break;
      }
      case ReadNodesViaContextEdgesConfig::CONTEXTS_BY_ARTIFACT: {
        InitializeReadRequest<GetContextsByArtifactRequest>(read_request);
        Artifact picked_node = absl::get<Artifact>(existing_nodes[node_index]);
        absl::get<GetContextsByArtifactRequest>(read_request)
            .set_artifact_id(picked_node.id());
        break;
      }
      case ReadNodesViaContextEdgesConfig::CONTEXTS_BY_EXECUTION: {
        InitializeReadRequest<GetContextsByExecutionRequest>(read_request);
        Execution picked_node =
            absl::get<Execution>(existing_nodes[node_index]);
        absl::get<GetContextsByExecutionRequest>(read_request)
            .set_execution_id(picked_node.id());
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for ReadNodesViaContextEdges!";
    }
    work_items_.emplace_back(read_request, curr_bytes);
  }

  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadNodesViaContextEdges::RunOpImpl(
    const int64 work_items_index, MetadataStore* store) {
  switch (read_nodes_via_context_edges_config_.specification()) {
    case ReadNodesViaContextEdgesConfig::ARTIFACTS_BY_CONTEXT: {
      GetArtifactsByContextRequest request =
          absl::get<GetArtifactsByContextRequest>(
              work_items_[work_items_index].first);
      GetArtifactsByContextResponse response;
      return store->GetArtifactsByContext(request, &response);
      break;
    }
    case ReadNodesViaContextEdgesConfig::EXECUTIONS_BY_CONTEXT: {
      GetExecutionsByContextRequest request =
          absl::get<GetExecutionsByContextRequest>(
              work_items_[work_items_index].first);
      GetExecutionsByContextResponse response;
      return store->GetExecutionsByContext(request, &response);
      break;
    }
    case ReadNodesViaContextEdgesConfig::CONTEXTS_BY_ARTIFACT: {
      GetContextsByArtifactRequest request =
          absl::get<GetContextsByArtifactRequest>(
              work_items_[work_items_index].first);
      GetContextsByArtifactResponse response;
      return store->GetContextsByArtifact(request, &response);
      break;
    }
    case ReadNodesViaContextEdgesConfig::CONTEXTS_BY_EXECUTION: {
      GetContextsByExecutionRequest request =
          absl::get<GetContextsByExecutionRequest>(
              work_items_[work_items_index].first);
      GetContextsByExecutionResponse response;
      return store->GetContextsByExecution(request, &response);
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status ReadNodesViaContextEdges::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string ReadNodesViaContextEdges::GetName() { return name_; }

}  // namespace ml_metadata
