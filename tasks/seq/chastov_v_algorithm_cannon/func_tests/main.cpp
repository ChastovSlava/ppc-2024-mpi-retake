// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/chastov_v_algorithm_cannon/include/ops_seq.hpp"

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

namespace {
std::vector<double> Multiplication(const std::vector<double> &matrix1, const std::vector<double> &matrix2,
                                   size_t size) {
  std::vector<double> result(size * size, 0.0);

  const double *ptr1 = matrix1.data();
  const double *ptr2 = matrix2.data();
  double *result_ptr = result.data();

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < size; ++k) {
        sum += ptr1[(i * size) + k] * ptr2[(k * size) + j];
      }
      result_ptr[(i * size) + j] = sum;
    }
  }

  return result;
}
}  // namespace

TEST(chastov_v_algorithm_cannon_seq, test_empty) {
  constexpr size_t kMatrix = 1;

  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2;
  std::vector<double> result_matrix(kMatrix * kMatrix, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size() + 1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_matrix.data()));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(chastov_v_algorithm_cannon_seq, test_multiplication_1x1) {
  constexpr size_t kMatrix = 1;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix);
  std::vector<double> expected_result = Multiplication(matrix1, matrix2, kMatrix);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(expected_result.begin(), expected_result.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(chastov_v_algorithm_cannon_seq, test_zero_matrix) {
  constexpr size_t kMatrix = 3;
  std::vector<double> matrix1(kMatrix * kMatrix, 0.0);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix);
  std::vector<double> expected(kMatrix * kMatrix, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(chastov_v_algorithm_cannon_seq, test_wrong_size) {
  constexpr size_t kMatrix = 10;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size() + 1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(chastov_v_algorithm_cannon_seq, test_reverse_matrix) {
  constexpr size_t kMatrix = 3;
  std::vector<double> iden{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> matrix1{2, 1, 1, 1, 3, 2, 1, 0, 0};
  std::vector<double> matrix2{0, 0, 1, -2, 1, 3, 3, -1, -5};
  std::vector<double> result_matrix(kMatrix * kMatrix);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(iden.begin(), iden.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(chastov_v_algorithm_cannon_seq, test_negative_numbers) {
  constexpr size_t kMatrix = 2;
  std::vector<double> matrix1{1, -2, -3, 4};
  std::vector<double> matrix2{-1, 2, 3, -4};
  std::vector<double> result_matrix(kMatrix * kMatrix);
  std::vector<double> expected{-7, 10, 15, -22};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(chastov_v_algorithm_cannon_seq, test_random_10x10) {
  constexpr size_t kMatrix = 10;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix);
  std::vector<double> expected_result = Multiplication(matrix1, matrix2, kMatrix);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(expected_result.begin(), expected_result.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(chastov_v_algorithm_cannon_seq, test_random_100x100) {
  constexpr size_t kMatrix = 100;
  std::vector<double> matrix1 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> matrix2 = GenerationRandVector(kMatrix * kMatrix);
  std::vector<double> result_matrix(kMatrix * kMatrix);
  std::vector<double> expected_result = Multiplication(matrix1, matrix2, kMatrix);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(expected_result.begin(), expected_result.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(chastov_v_algorithm_cannon_seq, test_diagonal_matrix) {
  constexpr size_t kMatrix = 3;
  std::vector<double> matrix1{2, 0, 0, 0, 3, 0, 0, 0, 4};
  std::vector<double> matrix2{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> result_matrix(kMatrix * kMatrix);
  std::vector<double> expected{2, 4, 6, 12, 15, 18, 28, 32, 36};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
  task_data_seq->inputs_count.emplace_back(matrix1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
  task_data_seq->inputs_count.emplace_back(matrix2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<size_t *>(&kMatrix)));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
  task_data_seq->outputs_count.emplace_back(result_matrix.size());

  chastov_v_algorithm_cannon_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), result_matrix.begin(),
                         [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}