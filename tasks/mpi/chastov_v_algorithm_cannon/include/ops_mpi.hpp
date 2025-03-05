// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chastov_v_algorithm_cannon_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  size_t matrix_size_{}, total_elements_{};
  std::vector<double> first_matrix_, second_matrix_, result_matrix_;
  std::vector<double> block_1_, block_2_, local_c_;
  boost::mpi::communicator world_;

  bool PrepareComputation(boost::mpi::communicator& sub_world, int& submatrix_size, int& block_size);
  bool InitializeBlocks(boost::mpi::communicator& sub_world, int submatrix_size, int block_size);
  bool CommunicateAndCompute(boost::mpi::communicator& sub_world, int submatrix_size, int block_size);
  bool ShiftBlocks(boost::mpi::communicator& sub_world, int submatrix_size, int block_size);
  bool ComputeAndGather(boost::mpi::communicator& sub_world, int submatrix_size, int block_size);
};

}  // namespace chastov_v_algorithm_cannon_mpi