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
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

// Sets up and executes the given FillTypes workloads.
tensorflow::Status SetUpAndExecuteWorkload(
    MetadataStore* store, std::unique_ptr<FillTypes>& workload) {
  TF_RETURN_IF_ERROR(workload->SetUp(store));
  for (int64 i = 0; i < workload->num_operations(); ++i) {
    OpStats op_stats;
    TF_RETURN_IF_ERROR(workload->RunOp(i, store, op_stats));
  }
  return tensorflow::Status::OK();
}

// Checks types update status. If the updates are working properly, the type id
// should remain the same even after the updates. On the other hand, the
// properties size for each type should be greater than before since some new
// fields have been added in the update process.
template <typename T>
bool CheckTypesUpdateStatus(const Type& type_before, const Type& type_after) {
  if (absl::get<T>(type_before).id() != absl::get<T>(type_after).id()) {
    return false;
  }
  if (absl::get<T>(type_before).properties().size() >=
      absl::get<T>(type_after).properties().size()) {
    return false;
  }
  return true;
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillTypes insert tests.
// The parameter here is the specific Workload configurations that
// ensures test coverage.
class FillTypesInsertParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_types_insert_ = absl::make_unique<FillTypes>(
        FillTypes(GetParam().fill_types_config(), GetParam().num_operations()));
  }

  std::unique_ptr<FillTypes> fill_types_insert_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes insert cases.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_P(FillTypesInsertParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(fill_types_insert_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_types_insert_->num_operations());
}

// Tests the RunOpImpl() for FillTypes insert cases.
// Checks indeed all the work items have been executed and the number of the
// types inside db is the same as the number of operations specified in the
// workload.
TEST_P(FillTypesInsertParameterizedTestFixture, InsertTest) {
  TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_insert_));
  // Gets all the existing current types inside db for later update cases.
  std::vector<Type> existing_types;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types));
  EXPECT_EQ(GetParam().num_operations(), existing_types.size());
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillTypes update tests.
// The parameter here is the pair of specific WorkloadConfig that contains
// insert and update FillTypes configurations which can ensure test
// coverage.
class FillTypesUpdateParameterizedTestFixture
    : public ::testing::TestWithParam<
          std::pair<WorkloadConfig, WorkloadConfig>> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    // Inserts some types into the db for later update since the test db is
    // empty in the beginning. `GetParam()` will return a pair of insert and
    // update workload configuration, so `GetParam().first` is the Workload
    // configuration contains insert FillTypes configurations.
    fill_types_insert_ = absl::make_unique<FillTypes>(
        FillTypes(GetParam().first.fill_types_config(),
                  GetParam().first.num_operations()));
    TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_insert_));
  }

  std::unique_ptr<FillTypes> fill_types_insert_;
  std::unique_ptr<FillTypes> fill_types_update_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes update cases. Checks the SetUpImpl()
// indeed prepares a list of work items whose length is the same as the
// specified number of operations.
TEST_P(FillTypesUpdateParameterizedTestFixture, SetUpImplTest) {
  // `GetParam().second` is the Workload configuration contains update FillTypes
  // configurations.
  fill_types_update_ = absl::make_unique<FillTypes>(
      FillTypes(GetParam().second.fill_types_config(),
                GetParam().second.num_operations()));
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().second.num_operations(),
            fill_types_update_->num_operations());
}

// Tests the RunOpImpl() for FillTypes update cases.
// Checks indeed all the work items have been executed and there are certain
// number of existed types inside db have been updated.
TEST_P(FillTypesUpdateParameterizedTestFixture, UpdateTest) {
  // Gets the existing types in db before update for later comparison.
  std::vector<Type> existing_types_before_update;
  // Passes the `GetParam().second.fill_types_config()` instead of
  // `GetParam().first.fill_types_config()` because we will only interested in
  // the same kind of types inside db as specified by the update workload
  // configuration.
  TF_ASSERT_OK(GetExistingTypes(GetParam().second.fill_types_config(),
                                store_.get(), existing_types_before_update));

  fill_types_update_ = absl::make_unique<FillTypes>(
      FillTypes(GetParam().second.fill_types_config(),
                GetParam().second.num_operations()));

  // Sets up and executes the update workloads.
  TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_update_));

  // Gets the existing types in db after update for later comparison.
  std::vector<Type> existing_types_after_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().second.fill_types_config(),
                                store_.get(), existing_types_after_update));

  // For the first `min(GetParam().second.num_operations(),
  // existing_types_before_update.size())` types inside db, uses
  // CheckTypesUpdateStatus() to check their update status.
  // For normal update cases where the number of types existed before update is
  // greater or equal than the number of operations for update, the smaller one
  // will be GetParam().second.num_operations().
  // On the other hand, if the number of types existed before update is less
  // than the number of operations of update where making up and inserting new
  // types is needed, the smaller one will be
  // existing_types_before_update.size(). So, we only check the the first
  // `min(GetParam().second.num_operations(),
  // existing_types_before_update.size())` types inside db for their update
  // status as they both exist before and after the updates.
  for (int64 i = 0; i < std::min((int64)GetParam().second.num_operations(),
                                 (int64)existing_types_before_update.size());
       ++i) {
    switch (GetParam().second.fill_types_config().specification()) {
      case FillTypesConfig::ARTIFACT_TYPE: {
        CheckTypesUpdateStatus<ArtifactType>(existing_types_before_update[i],
                                             existing_types_after_update[i]);
        break;
      }
      case FillTypesConfig::EXECUTION_TYPE: {
        CheckTypesUpdateStatus<ExecutionType>(existing_types_before_update[i],
                                              existing_types_after_update[i]);
        break;
      }
      case FillTypesConfig::CONTEXT_TYPE: {
        CheckTypesUpdateStatus<ContextType>(existing_types_before_update[i],
                                            existing_types_after_update[i]);
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillTypes configuration input "
                      "in testing !";
    }
  }

  // The total number of types inside db should be the bigger one between
  // existing_types_before_update.size() and number of operations for update.
  // For normal update cases, it will be existing_types_before_update.size().
  // For make up update cases where new types have been inserted into the db, it
  // will be number of operations for update.
  EXPECT_EQ((int64)existing_types_after_update.size(),
            std::max((int64)GetParam().second.num_operations(),
                     (int64)existing_types_before_update.size()));
}  // namespace

INSTANTIATE_TEST_CASE_P(
    FillTypesInsertTest, FillTypesInsertParameterizedTestFixture,
    ::testing::Values(testing::ParseTextProtoOrDie<WorkloadConfig>(
                          R"(
                            fill_types_config: {
                              update: false
                              specification: ARTIFACT_TYPE
                              num_properties { minimum: 1 maximum: 10 }
                            }
                            num_operations: 100
                          )"),
                      testing::ParseTextProtoOrDie<WorkloadConfig>(
                          R"(
                            fill_types_config: {
                              update: false
                              specification: EXECUTION_TYPE
                              num_properties { minimum: 1 maximum: 10 }
                            }
                            num_operations: 100
                          )"),
                      testing::ParseTextProtoOrDie<WorkloadConfig>(
                          R"(
                            fill_types_config: {
                              update: false
                              specification: CONTEXT_TYPE
                              num_properties { minimum: 1 maximum: 10 }
                            }
                            num_operations: 100
                          )")));

INSTANTIATE_TEST_CASE_P(
    FillTypesUpdateTest, FillTypesUpdateParameterizedTestFixture,
    ::testing::Values(
        std::make_pair(testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: false
                               specification: ARTIFACT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 100
                           )"),
                       testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: true
                               specification: ARTIFACT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 50
                           )")),
        std::make_pair(testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: false
                               specification: ARTIFACT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 50
                           )"),
                       testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: true
                               specification: ARTIFACT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 100
                           )")),
        std::make_pair(testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: false
                               specification: EXECUTION_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 100
                           )"),
                       testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: true
                               specification: EXECUTION_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 50
                           )")),
        std::make_pair(testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: false
                               specification: EXECUTION_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 50
                           )"),
                       testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: true
                               specification: EXECUTION_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 100
                           )")),
        std::make_pair(testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: false
                               specification: CONTEXT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 100
                           )"),
                       testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: true
                               specification: CONTEXT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 50
                           )")),
        std::make_pair(testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: false
                               specification: CONTEXT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 50
                           )"),
                       testing::ParseTextProtoOrDie<WorkloadConfig>(
                           R"(
                             fill_types_config: {
                               update: true
                               specification: CONTEXT_TYPE
                               num_properties { minimum: 1 maximum: 10 }
                             }
                             num_operations: 100
                           )"))));

}  // namespace
}  // namespace ml_metadata
