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
#include "ml_metadata/tools/mlmd_bench/fill_nodes_workload.h"

#include <random>

#include "ml_metadata/metadata_store/metadata_store.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

// Defines a Type can be ArtifactType / ExecutionType / ContextType.
using Type = absl::variant<ArtifactType, ExecutionType, ContextType>;

// Gets all the existing types (the specific types that indicated by
// `fill_types_config`) inside db and store them into `existing_types`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingTypes(const FillNodesConfig& fill_nodes_config,
                                    MetadataStore* store,
                                    std::vector<Type>& existing_types) {
  switch (fill_nodes_config.specification()) {
    case FillNodesConfig::ARTIFACT: {
      GetArtifactTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetArtifactTypes(
          /*request=*/{}, &get_response));
      for (auto& artifact_type : get_response.artifact_types()) {
        existing_types.push_back(artifact_type);
      }
      break;
    }
    case FillNodesConfig::EXECUTION: {
      GetExecutionTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetExecutionTypes(
          /*request=*/{}, &get_response));
      for (auto& execution_type : get_response.execution_types()) {
        existing_types.push_back(execution_type);
      }
      break;
    }
    case FillNodesConfig::CONTEXT: {
      GetContextTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetContextTypes(
          /*request=*/{}, &get_response));
      for (auto& context_type : get_response.context_types()) {
        existing_types.push_back(context_type);
      }
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillTypes!";
  }
  return tensorflow::Status::OK();
}

void InitializePutRequest(const FillNodesConfig& fill_nodes_config,
                          FillNodesWorkItemType& put_request) {
  switch (fill_nodes_config.specification()) {
    case FillNodesConfig::ARTIFACT: {
      put_request.emplace<PutArtifactsRequest>();
      break;
    }
    case FillNodesConfig::EXECUTION: {
      put_request.emplace<PutExecutionsRequest>();
      break;
    }
    case FillNodesConfig::CONTEXT: {
      put_request.emplace<PutContextsRequest>();
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillNodes!";
  }
}

template <typename T, typename NT>
void GenerateNode(const int64 num_properties, const int64 string_value_bytes,
                  const T& type, NT& node, int64& curr_bytes) {
  node.set_type_id(type.id());
  int64 curr_num_properties = 0;
  auto it = type.properties().begin();
  while (curr_num_properties < num_properties &&
         it != type.properties().end()) {
    std::string value(string_value_bytes, '*');
    (*node.mutable_properties())[it->first].set_string_value(value);
    curr_num_properties++;
    it++;
  }
  while (curr_num_properties < num_properties) {
    std::string value(string_value_bytes, '*');
    (*node.mutable_custom_properties())[absl::StrCat("custom_p-",
                                                     curr_num_properties)]
        .set_string_value(value);
    curr_num_properties++;
  }
}

}  // namespace

FillNodes::FillNodes(const FillNodesConfig& fill_nodes_config,
                     int64 num_operations)
    : fill_nodes_config_(fill_nodes_config), num_operations_(num_operations) {
  switch (fill_nodes_config_.specification()) {
    case FillNodesConfig::ARTIFACT: {
      name_ = "fill_artifact";
      break;
    }
    case FillNodesConfig::EXECUTION: {
      name_ = "fill_execution";
      break;
    }
    case FillNodesConfig::CONTEXT: {
      name_ = "fill_context";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillNodes!";
  }
  if (fill_nodes_config_.update()) {
    name_ += "(update)";
  }
}

tensorflow::Status FillNodes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  UniformDistribution num_properties_dist = fill_nodes_config_.num_properties();
  std::uniform_int_distribution<int64> uniform_dist_properties{
      num_properties_dist.minimum(), num_properties_dist.maximum()};

  UniformDistribution string_value_bytes_dist =
      fill_nodes_config_.string_value_bytes();
  std::uniform_int_distribution<int64> uniform_dist_string{
      string_value_bytes_dist.minimum(), string_value_bytes_dist.maximum()};

  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(
      GetExistingTypes(fill_nodes_config_, store, existing_types));

  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    FillNodesWorkItemType put_request;
    InitializePutRequest(fill_nodes_config_, put_request);
    const int64 num_properties = uniform_dist_properties(gen);
    const int64 string_value_bytes = uniform_dist_string(gen);
    const int64 type_index = uniform_dist_type_index(gen);
    switch (fill_nodes_config_.specification()) {
      case FillNodesConfig::ARTIFACT: {
        GenerateNode<ArtifactType, Artifact>(
            num_properties, string_value_bytes,
            absl::get<ArtifactType>(existing_types[type_index]),
            *(absl::get<PutArtifactsRequest>(put_request).add_artifacts()),
            curr_bytes);
        break;
      }
      case FillNodesConfig::EXECUTION: {
        GenerateNode<ExecutionType, Execution>(
            num_properties, string_value_bytes,
            absl::get<ExecutionType>(existing_types[type_index]),
            *(absl::get<PutExecutionsRequest>(put_request).add_executions()),
            curr_bytes);
        break;
      }
      case FillNodesConfig::CONTEXT: {
        GenerateNode<ContextType, Context>(
            num_properties, string_value_bytes,
            absl::get<ContextType>(existing_types[type_index]),
            *(absl::get<PutContextsRequest>(put_request).add_contexts()),
            curr_bytes);
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillNodes!";
    }
    work_items_.emplace_back(put_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status FillNodes::RunOpImpl(const int64 work_items_index,
                                        MetadataStore* store) {
  switch (fill_nodes_config_.specification()) {
    case FillNodesConfig::ARTIFACT: {
      PutArtifactsRequest put_request =
          absl::get<PutArtifactsRequest>(work_items_[work_items_index].first);
      PutArtifactsResponse put_response;
      TF_RETURN_IF_ERROR(store->PutArtifacts(put_request, &put_response));
      return tensorflow::Status::OK();
    }
    case FillNodesConfig::EXECUTION: {
      PutExecutionsRequest put_request =
          absl::get<PutExecutionsRequest>(work_items_[work_items_index].first);
      PutExecutionsResponse put_response;
      TF_RETURN_IF_ERROR(store->PutExecutions(put_request, &put_response));
      return tensorflow::Status::OK();
    }
    case FillNodesConfig::CONTEXT: {
      PutContextsRequest put_request =
          absl::get<PutContextsRequest>(work_items_[work_items_index].first);
      PutContextsResponse put_response;
      TF_RETURN_IF_ERROR(store->PutContexts(put_request, &put_response));
      return tensorflow::Status::OK();
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
  return tensorflow::errors::InvalidArgument(
      "Cannot execute the query due to wrong specification!");
}

tensorflow::Status FillNodes::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillNodes::GetName() { return name_; }

}  // namespace ml_metadata
