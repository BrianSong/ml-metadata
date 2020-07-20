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

// PutEventsRequest put_events_request = ParseTextProtoOrDie<PutEventsRequest>(
//     R"(
//       events: {}
//     )");
// put_events_request.mutable_events(0)->set_artifact_id(
//     put_artifacts_response.artifact_ids(0));
// put_events_request.mutable_events(0)->set_execution_id(
//     put_executions_response.execution_ids(0));
// put_events_request.mutable_events(0)->set_type(Event::DECLARED_OUTPUT);
// PutEventsResponse put_events_response;
// TF_ASSERT_OK(metadata_store_->PutEvents(put_events_request,
//                                         &put_events_response));
namespace ml_metadata {
namespace {}  // namespace

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
  return tensorflow::Status::OK();
}

tensorflow::Status FillEvents::RunOpImpl(const int64 work_items_index,
                                         MetadataStore* store) {
  return tensorflow::Status::OK();
}

tensorflow::Status FillEvents::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillEvents::GetName() { return name_; }

}  // namespace ml_metadata
