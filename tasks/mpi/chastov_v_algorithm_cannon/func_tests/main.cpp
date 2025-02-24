// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/chastov_v_algorithm_cannon/include/ops_mpi.hpp"

namespace {
bool CompareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon = 1e-9);
}  // namespace

namespace {
std::vector<double> GenerationRandVector(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  std::vector<double> vec(size);
  std::ranges::generate(vec, [&gen, &dist]() { return dist(gen); });
  return vec;
}
}  // namespace

TEST(chastov_v_algorithm_cannon_mpi, test_empty) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 3;
  std::vector<double> matrix1;
  std::vector<double> matrix2;
  std::vector<double> result_matrix;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix2 = GenerationRandVector(kMatrix * kMatrix);
    result_matrix = std::vector<double>(kMatrix * kMatrix, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), false);
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_wrong_size) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 3;
  std::vector<double> matrix1;
  std::vector<double> matrix2;
  std::vector<double> result_matrix;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = GenerationRandVector(kMatrix * kMatrix);
    matrix2 = GenerationRandVector(kMatrix * kMatrix);
    result_matrix = std::vector<double>(kMatrix * kMatrix, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size() - 1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), false);
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_inverse) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 5;
  std::vector<double> iden = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<double> matrix1 = {-0.5, -0.5, 0,    0.5, 0,    0.5, 0,   -0.25, 0, -0.5, 1.5, 0.5, -0.25,
                                 -0.5, 0,    -0.5, 0.5, 0.25, 0,   0.5, -0.5,  0, 0.25, 0,   0};
  std::vector<double> matrix2 = {2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 4, 0, 4, 0, 8, 4, 2, 2, 2, 2, 0, -2, 0, 0, -2};
  std::vector<double> result_matrix;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    result_matrix = std::vector<double>(kMatrix * kMatrix, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_TRUE(CompareMatrices(iden, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_iden) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 5;
  std::vector<double> matrix1 = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<double> matrix2;
  std::vector<double> result_matrix;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix2 = GenerationRandVector(kMatrix * kMatrix);
    result_matrix = std::vector<double>(kMatrix * kMatrix, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_TRUE(CompareMatrices(matrix2, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_multiplication_1x1) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 1;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    double expected = matrix1[0] * matrix2[0];
    ASSERT_NEAR(result_matrix[0], expected, 1e-9);
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_zero_matrix) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 2;
  std::vector<double> matrix1(kMatrix * kMatrix, 0.0);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected(kMatrix * kMatrix, 0.0);
    ASSERT_TRUE(CompareMatrices(expected, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_reverse_matrix) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 2;
  std::vector<double> matrix1 = {-1.0, -2.0, -3.0, -4.0};
  std::vector<double> matrix2 = {-5.0, -6.0, -7.0, -8.0};
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected = {19.0, 22.0, 43.0, 50.0};
    ASSERT_TRUE(CompareMatrices(expected, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_negative_numbers) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 2;
  std::vector<double> matrix1 = {-1.0, 2.0, 3.0, -4.0};
  std::vector<double> matrix2 = {5.0, -6.0, -7.0, 8.0};
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected = {-19.0, 22.0, 43.0, -50.0};
    ASSERT_TRUE(CompareMatrices(expected, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_random_10x10) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 10;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected(kMatrix * kMatrix);
    for (size_t i = 0; i < kMatrix; ++i) {
      for (size_t j = 0; j < kMatrix; ++j) {
        expected[(i * kMatrix) + j] = 0.0;
        for (size_t k = 0; k < kMatrix; ++k) {
          expected[(i * kMatrix) + j] += matrix1[(i * kMatrix) + k] * matrix2[(k * kMatrix) + j];
        }
      }
    }
    ASSERT_TRUE(CompareMatrices(expected, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_random_100x100) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 100;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected(kMatrix * kMatrix);
    for (size_t i = 0; i < kMatrix; ++i) {
      for (size_t j = 0; j < kMatrix; ++j) {
        expected[(i * kMatrix) + j] = 0.0;
        for (size_t k = 0; k < kMatrix; ++k) {
          expected[(i * kMatrix) + j] += matrix1[(i * kMatrix) + k] * matrix2[(k * kMatrix) + j];
        }
      }
    }
    ASSERT_TRUE(CompareMatrices(expected, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_diagonal_matrix) {
  boost::mpi::communicator world;
  constexpr size_t kMatrix = 3;
  std::vector<double> matrix1 = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  chastov_v_algorithm_cannon_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected(kMatrix * kMatrix);
    for (size_t i = 0; i < kMatrix; ++i) {
      for (size_t j = 0; j < kMatrix; ++j) {
        expected[(i * kMatrix) + j] = matrix1[(i * kMatrix) + i] * matrix2[(i * kMatrix) + j];
      }
    }
    ASSERT_TRUE(CompareMatrices(expected, result_matrix));
  }
}

namespace {
bool CompareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon) {
  if (mat1.size() != mat2.size()) {
    return false;
  }
  for (size_t i = 0; i < mat1.size(); ++i) {
    if (std::abs(mat1[i] - mat2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}
}  // namespace