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
  int calculateBlockSize(int size);
  MPI_Comm createSubCommunicator(int rank, int block_size);
  void distributeMatrixBlocks(const std::vector<double>& matrix, std::vector<double>& temp_vec, int block_size,
                              int submatrix_size);
  bool performCannonAlgorithm(boost::mpi::communicator& sub_world, std::vector<double>& block_1,
                              std::vector<double>& block_2, std::vector<double>& block_c, int block_size,
                              int submatrix_size);
  void shiftBlocks(boost::mpi::communicator& sub_world, std::vector<double>& block, int send_rank, int recv_rank);
  void multiplyBlocks(const std::vector<double>& block_1, const std::vector<double>& block_2,
                      std::vector<double>& block_c, int submatrix_size);
  void assembleResultMatrix(const std::vector<double>& collected_vec, std::vector<double>& result_matrix,
                            int block_size, int submatrix_size);

 private:
  size_t matrix_size_{}, total_elements_{};
  std::vector<double> first_matrix_, second_matrix_, result_matrix_;
  boost::mpi::communicator world_;
};

}  // namespace chastov_v_algorithm_cannon_mpi