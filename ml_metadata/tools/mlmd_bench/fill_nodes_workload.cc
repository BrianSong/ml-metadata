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

tensorflow::Status GetTypes(const FillNodesConfig& fill_nodes_config,
                            MetadataStore* store,
                            GetTypesResponseType& get_response,
                            int64& num_types) {
  switch (fill_nodes_config.specification()) {
    case FillNodesConfig::ARTIFACT: {
      get_response.emplace<GetArtifactTypesResponse>();
      TF_RETURN_IF_ERROR(store->GetArtifactTypes(
          /*request=*/{}, &absl::get<GetArtifactTypesResponse>(get_response)));
      num_types = absl::get<GetArtifactTypesResponse>(get_response)
                      .artifact_types_size();
      break;
    }
    case FillNodesConfig::EXECUTION: {
      get_response.emplace<GetExecutionTypesResponse>();
      TF_RETURN_IF_ERROR(store->GetExecutionTypes(
          /*request=*/{}, &absl::get<GetExecutionTypesResponse>(get_response)));
      num_types = absl::get<GetExecutionTypesResponse>(get_response)
                      .execution_types_size();
      break;
    }
    case FillNodesConfig::CONTEXT: {
      get_response.emplace<GetContextTypesResponse>();
      TF_RETURN_IF_ERROR(store->GetContextTypes(
          /*request=*/{}, &absl::get<GetContextTypesResponse>(get_response)));
      num_types =
          absl::get<GetContextTypesResponse>(get_response).context_types_size();
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillNodes!";
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
  int64 num_types = 0;

  UniformDistribution num_properties_dist = fill_nodes_config_.num_properties();
  std::uniform_int_distribution<int64> uniform_dist_properties{
      num_properties_dist.minimum(), num_properties_dist.maximum()};

  UniformDistribution string_value_bytes_dist =
      fill_nodes_config_.string_value_bytes();
  std::uniform_int_distribution<int64> uniform_dist_string{
      string_value_bytes_dist.minimum(), string_value_bytes_dist.maximum()};

  std::uniform_int_distribution<int64> uniform_dist_type_index{0,
                                                               num_types - 1};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  GetTypesResponseType get_response;
  TF_RETURN_IF_ERROR(
      GetTypes(fill_nodes_config_, store, get_response, num_types));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    FillNodesWorkItemType put_request;
    InitializePutRequest(fill_nodes_config_, put_request);
    const int64 num_properties = uniform_dist_properties(gen);
    const int64 string_value_bytes = uniform_dist_string(gen);
    const int64 type_index = uniform_dist_type_index(gen);
    switch (fill_nodes_config_.specification()) {
      case FillNodesConfig::ARTIFACT: {
        break;
      }
      case FillNodesConfig::EXECUTION: {
        break;
      }
      case FillNodesConfig::CONTEXT: {
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillNodes!";
    }
  }
}

}  // namespace ml_metadata
