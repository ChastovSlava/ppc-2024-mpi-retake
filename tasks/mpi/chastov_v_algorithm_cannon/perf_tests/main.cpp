// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chastov_v_algorithm_cannon/include/ops_mpi.hpp"

static bool compareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon = 1e-9);

TEST(chastov_v_algorithm_cannon_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int kMatrix = 500;

  // Create data
  std::vector<double> matrix1;
  std::vector<double> matrix2;
  std::vector<double> resultMatrix;

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = std::vector<double>(kMatrix * kMatrix, 0.0);
    matrix2 = std::vector<double>(kMatrix * kMatrix, 1.0);
    resultMatrix = std::vector<double>(kMatrix * kMatrix);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&kMatrix));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resultMatrix));
    task_data_mpi->outputs_count.emplace_back(resultMatrix.size());
  }

  // Create Task
  auto test_task_mpi = std::make_shared<chastov_v_algorithm_cannon_mpi::TestTaskMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(compareMatrices(matrix1, resultMatrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_task_run) {
  boost::mpi::communicator world;
  int kMatrix = 500;

  // Create data
  std::vector<double> matrix1;
  std::vector<double> matrix2;
  std::vector<double> resultMatrix;

  // Create TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = std::vector<double>(kMatrix * kMatrix, 0.0);
    matrix2 = std::vector<double>(kMatrix * kMatrix, 1.0);
    resultMatrix = std::vector<double>(kMatrix * kMatrix, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&kMatrix));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resultMatrix));
    task_data_mpi->outputs_count.emplace_back(resultMatrix.size());
  }

  auto test_task_mpi = std::make_shared<chastov_v_algorithm_cannon_mpi::TestTaskMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(compareMatrices(matrix1, resultMatrix));
  }
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