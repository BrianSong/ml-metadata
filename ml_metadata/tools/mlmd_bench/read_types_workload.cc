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
#include "ml_metadata/tools/mlmd_bench/read_types_workload.h"

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
void InitializeReadRequest(ReadTypesWorkItemType& read_request) {
  read_request.emplace<T>();
}

// Calculates the transferred bytes for each types that will be inserted or
// updated later.
template <typename T>
tensorflow::Status GetTransferredBytes(const T& type, int64& curr_bytes) {
  curr_bytes += type.name().size();
  for (auto& pair : type.properties()) {
    // Includes the bytes for properties' name size.
    curr_bytes += pair.first.size();
    // Includes the bytes for properties' value enumeration size.
    if (pair.second == PropertyType::UNKNOWN) {
      return tensorflow::errors::InvalidArgument("Invalid PropertyType!");
    }
    // As we uses a TINYINT to store the enum.
    curr_bytes += 1;
  }
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status GetTransferredBytesForAllTypes(
    std::vector<Type>& existing_types, int64& curr_bytes) {
  for (auto& type : existing_types) {
    TF_RETURN_IF_ERROR(GetTransferredBytes<T>(absl::get<T>(type), curr_bytes));
  }
  return tensorflow::Status::OK();
}

}  // namespace

ReadTypes::ReadTypes(const ReadTypesConfig& read_types_config,
                     const int64 num_operations)
    : read_types_config_(read_types_config), num_operations_(num_operations) {
  switch (read_types_config_.specification()) {
    case ReadTypesConfig::ALL_ARTIFACT_TYPES: {
      name_ = "read_all_artifact_types";
      break;
    }
    case ReadTypesConfig::ALL_EXECUTION_TYPES: {
      name_ = "read_all_execution_types";
      break;
    }
    case ReadTypesConfig::ALL_CONTEXT_TYPES: {
      name_ = "read_all_context_types";
      break;
    }
    case ReadTypesConfig::ARTIFACT_TYPES_BY_ID: {
      name_ = "read_artifact_types_by_id";
      break;
    }
    case ReadTypesConfig::EXECUTION_TYPES_BY_ID: {
      name_ = "read_execution_types_by_id";
      break;
    }
    case ReadTypesConfig::CONTEXT_TYPES_BY_ID: {
      name_ = "read_context_types_by_id";
      break;
    }
    case ReadTypesConfig::ARTIFACT_TYPE_BY_NAME: {
      name_ = "read_artifact_type_by_name";
      break;
    }
    case ReadTypesConfig::EXECUTION_TYPE_BY_NAME: {
      name_ = "read_execution_type_by_name";
      break;
    }
    case ReadTypesConfig::CONTEXT_TYPE_BY_NAME: {
      name_ = "read_context_type_by_name";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for ReadTypes!";
  }
}

tensorflow::Status ReadTypes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(GetExistingTypes(read_types_config_.specification() % 3,
                                      store, existing_types));

  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    ReadTypesWorkItemType read_request;
    const int64 type_index = uniform_dist_type_index(gen);
    switch (read_types_config_.specification()) {
      case ReadTypesConfig::ALL_ARTIFACT_TYPES: {
        InitializeReadRequest<GetArtifactTypesRequest>(read_request);
        TF_RETURN_IF_ERROR(GetTransferredBytesForAllTypes<ArtifactType>(
            existing_types, curr_bytes));
        break;
      }
      case ReadTypesConfig::ALL_EXECUTION_TYPES: {
        InitializeReadRequest<GetExecutionTypesRequest>(read_request);
        TF_RETURN_IF_ERROR(GetTransferredBytesForAllTypes<ExecutionType>(
            existing_types, curr_bytes));
        break;
      }
      case ReadTypesConfig::ALL_CONTEXT_TYPES: {
        InitializeReadRequest<GetContextTypesRequest>(read_request);
        TF_RETURN_IF_ERROR(GetTransferredBytesForAllTypes<ContextType>(
            existing_types, curr_bytes));
        break;
      }
      case ReadTypesConfig::ARTIFACT_TYPES_BY_ID: {
        InitializeReadRequest<GetArtifactTypesByIDRequest>(read_request);
        absl::get<GetArtifactTypesByIDRequest>(read_request)
            .add_type_ids(
                absl::get<ArtifactType>(existing_types[type_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes<ArtifactType>(
            absl::get<ArtifactType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::EXECUTION_TYPES_BY_ID: {
        InitializeReadRequest<GetExecutionTypesByIDRequest>(read_request);
        absl::get<GetExecutionTypesByIDRequest>(read_request)
            .add_type_ids(
                absl::get<ExecutionType>(existing_types[type_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes<ExecutionType>(
            absl::get<ExecutionType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::CONTEXT_TYPES_BY_ID: {
        InitializeReadRequest<GetContextTypesByIDRequest>(read_request);
        absl::get<GetContextTypesByIDRequest>(read_request)
            .add_type_ids(
                absl::get<ContextType>(existing_types[type_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes<ContextType>(
            absl::get<ContextType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::ARTIFACT_TYPE_BY_NAME: {
        InitializeReadRequest<GetArtifactTypeRequest>(read_request);
        absl::get<GetArtifactTypeRequest>(read_request)
            .set_type_name(
                absl::get<ArtifactType>(existing_types[type_index]).name());
        TF_RETURN_IF_ERROR(GetTransferredBytes<ArtifactType>(
            absl::get<ArtifactType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::EXECUTION_TYPE_BY_NAME: {
        InitializeReadRequest<GetExecutionTypeRequest>(read_request);
        absl::get<GetExecutionTypeRequest>(read_request)
            .set_type_name(
                absl::get<ExecutionType>(existing_types[type_index]).name());
        TF_RETURN_IF_ERROR(GetTransferredBytes<ExecutionType>(
            absl::get<ExecutionType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::CONTEXT_TYPE_BY_NAME: {
        InitializeReadRequest<GetContextTypeRequest>(read_request);
        absl::get<GetContextTypeRequest>(read_request)
            .set_type_name(
                absl::get<ContextType>(existing_types[type_index]).name());
        TF_RETURN_IF_ERROR(GetTransferredBytes<ContextType>(
            absl::get<ContextType>(existing_types[type_index]), curr_bytes));
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for ReadTypes!";
    }
    std::cout << curr_bytes << std::endl;
    work_items_.emplace_back(read_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadTypes::RunOpImpl(const int64 work_items_index,
                                        MetadataStore* store) {
  switch (read_types_config_.specification()) {
    case ReadTypesConfig::ALL_ARTIFACT_TYPES: {
      GetArtifactTypesRequest request = absl::get<GetArtifactTypesRequest>(
          work_items_[work_items_index].first);
      GetArtifactTypesResponse response;
      return store->GetArtifactTypes(request, &response);
      break;
    }
    case ReadTypesConfig::ALL_EXECUTION_TYPES: {
      GetExecutionTypesRequest request = absl::get<GetExecutionTypesRequest>(
          work_items_[work_items_index].first);
      GetExecutionTypesResponse response;
      return store->GetExecutionTypes(request, &response);
      break;
    }
    case ReadTypesConfig::ALL_CONTEXT_TYPES: {
      GetContextTypesRequest request = absl::get<GetContextTypesRequest>(
          work_items_[work_items_index].first);
      GetContextTypesResponse response;
      return store->GetContextTypes(request, &response);
      break;
    }
    case ReadTypesConfig::ARTIFACT_TYPES_BY_ID: {
      GetArtifactTypesByIDRequest request =
          absl::get<GetArtifactTypesByIDRequest>(
              work_items_[work_items_index].first);
      GetArtifactTypesByIDResponse response;
      return store->GetArtifactTypesByID(request, &response);
      break;
    }
    case ReadTypesConfig::EXECUTION_TYPES_BY_ID: {
      GetExecutionTypesByIDRequest request =
          absl::get<GetExecutionTypesByIDRequest>(
              work_items_[work_items_index].first);
      GetExecutionTypesByIDResponse response;
      return store->GetExecutionTypesByID(request, &response);
      break;
    }
    case ReadTypesConfig::CONTEXT_TYPES_BY_ID: {
      GetContextTypesByIDRequest request =
          absl::get<GetContextTypesByIDRequest>(
              work_items_[work_items_index].first);
      GetContextTypesByIDResponse response;
      return store->GetContextTypesByID(request, &response);
      break;
    }
    case ReadTypesConfig::ARTIFACT_TYPE_BY_NAME: {
      GetArtifactTypeRequest request = absl::get<GetArtifactTypeRequest>(
          work_items_[work_items_index].first);
      GetArtifactTypeResponse response;
      return store->GetArtifactType(request, &response);
      break;
    }
    case ReadTypesConfig::EXECUTION_TYPE_BY_NAME: {
      GetExecutionTypeRequest request = absl::get<GetExecutionTypeRequest>(
          work_items_[work_items_index].first);
      GetExecutionTypeResponse response;
      return store->GetExecutionType(request, &response);
      break;
    }
    case ReadTypesConfig::CONTEXT_TYPE_BY_NAME: {
      GetContextTypeRequest request =
          absl::get<GetContextTypeRequest>(work_items_[work_items_index].first);
      GetContextTypeResponse response;
      return store->GetContextType(request, &response);
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status ReadTypes::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string ReadTypes::GetName() { return name_; }

}  // namespace ml_metadata
