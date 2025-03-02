// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/chastov_v_algorithm_cannon/include/ops_seq.hpp"

static bool compareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon = 1e-9);

TEST(chastov_v_algorithm_cannon_seq, test_pipeline_run) {
  size_t kMatrix = 500;

  // Create data
  std::vector<double> matrix1(kMatrix * kMatrix, 0.0);
  std::vector<double> matrix2(kMatrix * kMatrix, 1.0);
  std::vector<double> resultMatrix(kMatrix * kMatrix);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&kMatrix));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resultMatrix));
  task_data_seq->outputs_count.emplace_back(resultMatrix.size());

  // Create Task
  auto testTaskSequential = std::make_shared<chastov_v_algorithm_cannon_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_TRUE(compareMatrices(matrix1, resultMatrix));
}

TEST(chastov_v_algorithm_cannon_seq, test_task_run) {
  size_t kMatrix = 500;

  // Create data
  std::vector<double> matrix1(kMatrix * kMatrix, 0.0);
  std::vector<double> matrix2(kMatrix * kMatrix, 1.0);
  std::vector<double> resultMatrix(kMatrix * kMatrix);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&kMatrix));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resultMatrix));
  task_data_seq->outputs_count.emplace_back(resultMatrix.size());

  // Create Task
  auto testTaskSequential = std::make_shared<chastov_v_algorithm_cannon_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_TRUE(compareMatrices(matrix1, resultMatrix));
}

static bool compareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon) {
  if (mat1.size() != mat2.size()) return false;
  for (size_t i = 0; i < mat1.size(); ++i) {
    if (std::abs(mat1[i] - mat2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}