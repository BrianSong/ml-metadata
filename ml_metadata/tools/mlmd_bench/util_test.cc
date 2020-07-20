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

#include <array>

#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int kNumberOfInsertedArtifactTypes = 51;
constexpr int kNumberOfInsertedExecutionTypes = 52;
constexpr int kNumberOfInsertedContextTypes = 53;

constexpr int kNumberOfInsertedArtifacts = 101;
constexpr int kNumberOfInsertedExecutions = 102;
constexpr int kNumberOfInsertedContexts = 103;

constexpr std::array<int, 3> get_specifications = {0, 1, 2};
constexpr std::array<int, 3> num_inserted_types = {
    kNumberOfInsertedArtifactTypes, kNumberOfInsertedExecutionTypes,
    kNumberOfInsertedContextTypes};
constexpr std::array<int, 3> num_inserted_nodes = {kNumberOfInsertedArtifacts,
                                                   kNumberOfInsertedExecutions,
                                                   kNumberOfInsertedContexts};

TEST(UtilInsertTest, InsertTypesTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, store.get()));

  GetArtifactTypesResponse get_artifact_types_response;
  TF_ASSERT_OK(store->GetArtifactTypes(
      /*request=*/{}, &get_artifact_types_response));
  GetExecutionTypesResponse get_execution_types_response;
  TF_ASSERT_OK(store->GetExecutionTypes(
      /*request=*/{}, &get_execution_types_response));
  GetContextTypesResponse get_context_types_response;
  TF_ASSERT_OK(store->GetContextTypes(
      /*request=*/{}, &get_context_types_response));

  EXPECT_EQ(kNumberOfInsertedArtifactTypes,
            get_artifact_types_response.artifact_types_size());
  EXPECT_EQ(kNumberOfInsertedExecutionTypes,
            get_execution_types_response.execution_types_size());
  EXPECT_EQ(kNumberOfInsertedContextTypes,
            get_context_types_response.context_types_size());
}

TEST(UtilInsertTest, InsertNodesTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  // Can I use InsertTypesInDb() here????????????????????????????????????
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, store.get()));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, store.get()));

  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(store->GetArtifacts(
      /*request=*/{}, &get_artifacts_response));
  GetExecutionsResponse get_executions_response;
  TF_ASSERT_OK(store->GetExecutions(
      /*request=*/{}, &get_executions_response));
  GetContextsResponse get_contexts_response;
  TF_ASSERT_OK(store->GetContexts(
      /*request=*/{}, &get_contexts_response));

  EXPECT_EQ(kNumberOfInsertedArtifacts,
            get_artifacts_response.artifacts_size());
  EXPECT_EQ(kNumberOfInsertedExecutions,
            get_executions_response.executions_size());
  EXPECT_EQ(kNumberOfInsertedContexts, get_contexts_response.contexts_size());
}

class UtilGetParameterizedTestFixture : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
  }

  std::unique_ptr<MetadataStore> store_;
};

TEST_P(UtilGetParameterizedTestFixture, GetTypesTest) {
  std::vector<Type> exisiting_types;
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, store_.get()));
  TF_ASSERT_OK(GetExistingTypes(GetParam(), store_.get(), exisiting_types));
  EXPECT_EQ(num_inserted_types[GetParam()], exisiting_types.size());
}

TEST_P(UtilGetParameterizedTestFixture, GetNodesTest) {
  std::vector<NodeType> exisiting_nodes_;
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, store_.get()));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, store_.get()));
  TF_ASSERT_OK(GetExistingNodes(GetParam(), store_.get(), exisiting_nodes_));
  EXPECT_EQ(num_inserted_nodes[GetParam()], exisiting_nodes_.size());
}

INSTANTIATE_TEST_CASE_P(UtilGetTest, UtilGetParameterizedTestFixture,
                        ::testing::ValuesIn(get_specifications));

}  // namespace
}  // namespace ml_metadata
