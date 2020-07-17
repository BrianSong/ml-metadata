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
#include "ml_metadata/tools/mlmd_bench/util.h"

#include <vector>

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

tensorflow::Status GetExistingTypes(const int specification,
                                    MetadataStore* store,
                                    std::vector<Type>& existing_types) {
  switch (specification) {
    // ArtifactType cases.
    case 0: {
      GetArtifactTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetArtifactTypes(
          /*request=*/{}, &get_response));
      for (auto& artifact_type : get_response.artifact_types()) {
        existing_types.push_back(artifact_type);
      }
      break;
    }
    // ExecutionType cases.
    case 1: {
      GetExecutionTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetExecutionTypes(
          /*request=*/{}, &get_response));
      for (auto& execution_type : get_response.execution_types()) {
        existing_types.push_back(execution_type);
      }
      break;
    }
    // ContextType cases.
    case 2: {
      GetContextTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetContextTypes(
          /*request=*/{}, &get_response));
      for (auto& context_type : get_response.context_types()) {
        existing_types.push_back(context_type);
      }
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for getting types in db!";
  }
  return tensorflow::Status::OK();
}

}  // namespace ml_metadata
