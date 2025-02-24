// Copyright 2024 Nesterov Alexander
#include "seq/chastov_v_algorithm_cannon/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

bool chastov_v_algorithm_cannon_seq::TestTaskSequential::PreProcessingImpl() {
  auto* first = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* second = reinterpret_cast<double*>(task_data->inputs[1]);

  first_matrix_ = std::vector<double>(first, first + total_elements_);
  second_matrix_ = std::vector<double>(second, second + total_elements_);

  result_matrix_.clear();
  result_matrix_.resize(total_elements_, 0.0);
  return true;
}

bool chastov_v_algorithm_cannon_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs[2] != nullptr) {
    matrix_size_ = *reinterpret_cast<size_t*>(task_data->inputs[2]);
  }

  total_elements_ = matrix_size_ * matrix_size_;

  bool valid = matrix_size_ > 0;
  valid &= task_data->inputs_count[2] == 1;
  valid &= task_data->inputs_count[0] == task_data->inputs_count[1];
  valid &= task_data->inputs_count[1] == total_elements_;
  valid &= task_data->inputs[0] != nullptr;
  valid &= task_data->inputs[1] != nullptr;
  valid &= task_data->outputs[0] != nullptr;
  valid &= static_cast<size_t>(task_data->outputs_count[0]) == total_elements_;

  return valid;
}

bool chastov_v_algorithm_cannon_seq::TestTaskSequential::RunImpl() {
  for (size_t i = 0; i < matrix_size_; ++i) {
    for (size_t j = 0; j < matrix_size_; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < matrix_size_; ++k) {
        sum += first_matrix_[(i * matrix_size_) + k] * second_matrix_[(k * matrix_size_) + j];
      }
      result_matrix_[(i * matrix_size_) + j] = sum;
    }
  }

  return true;
}

bool chastov_v_algorithm_cannon_seq::TestTaskSequential::PostProcessingImpl() {
  auto* output = reinterpret_cast<std::vector<double>*>(task_data->outputs[0]);
  output->assign(result_matrix_.begin(), result_matrix_.end());

  return true;
}