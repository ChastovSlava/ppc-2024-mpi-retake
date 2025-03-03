// Copyright 2023 Nesterov Alexander
#include "mpi/chastov_v_algorithm_cannon/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::PrepareComputation(boost::mpi::communicator& sub_world,
                                                                     int& submatrix_size, int& block_size) {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, total_elements_, 0);

  block_size = std::floor(std::sqrt(size));
  while (block_size > 0) {
    if (matrix_size_ % block_size == 0) {
      break;
    };
    --block_size;
  }
  block_size = std::max(block_size, 1);
  submatrix_size = static_cast<int>(matrix_size_ / block_size);

  int group_color = (rank < block_size * block_size) ? 1 : MPI_UNDEFINED;
  MPI_Comm sub_comm = MPI_COMM_NULL;
  MPI_Comm_split(world_, group_color, rank, &sub_comm);

  if (group_color == MPI_UNDEFINED) {
    return false;
  };

  sub_world = boost::mpi::communicator(sub_comm, boost::mpi::comm_take_ownership);
  return true;
}

bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* first = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* second = reinterpret_cast<double*>(task_data->inputs[1]);

    first_matrix_ = std::vector<double>(first, first + total_elements_);
    second_matrix_ = std::vector<double>(second, second + total_elements_);

    result_matrix_.clear();
    result_matrix_.resize(total_elements_, 0.0);
  }
  return true;
}

bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs[2] != nullptr) {
      matrix_size_ = reinterpret_cast<int*>(task_data->inputs[2])[0];
    }

    total_elements_ = matrix_size_ * matrix_size_;

    bool is_matrix_size_valid = matrix_size_ > 0;

    bool is_input_count_valid = task_data->inputs_count[2] == 1 &&
                                task_data->inputs_count[0] == task_data->inputs_count[1] &&
                                static_cast<int>(task_data->inputs_count[1]) == static_cast<int>(total_elements_);

    bool are_pointers_valid =
        task_data->inputs[0] != nullptr && task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;

    bool is_output_count_valid = static_cast<int>(task_data->outputs_count[0]) == static_cast<int>(total_elements_);

    return is_matrix_size_valid && is_input_count_valid && are_pointers_valid && is_output_count_valid;
  }

  return true;
}

// NOLINTBEGIN
bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::RunImpl() {
  boost::mpi::communicator sub_world;
  int block_size, submatrix_size;

  if (!PrepareComputation(sub_world, submatrix_size, block_size)) {
    return true;
  }

  int rank = sub_world.rank();
  int size = sub_world.size();

  std::vector<double> temp_vec_1(total_elements_);
  std::vector<double> temp_vec_2(total_elements_);
  if (rank == 0) {
    int index = 0;
    for (int block_row = 0; block_row < block_size; ++block_row) {
      for (int block_col = 0; block_col < block_size; ++block_col) {
        for (int i = 0; i < submatrix_size; ++i) {
          for (int j = 0; j < submatrix_size; ++j) {
            temp_vec_1[index + (i * submatrix_size) + j] =
                first_matrix_[((block_row * submatrix_size + i) * matrix_size_) + (block_col * submatrix_size + j)];
          }
        }

        for (int i = 0; i < submatrix_size; ++i) {
          for (int j = 0; j < submatrix_size; ++j) {
            temp_vec_2[index + (i * submatrix_size) + j] =
                second_matrix_[((block_row * submatrix_size + i) * matrix_size_) + (block_col * submatrix_size + j)];
          }
        }

        index += submatrix_size * submatrix_size;
      }
    }
  }

  std::vector<double> block_1(submatrix_size * submatrix_size);
  std::vector<double> block_2(submatrix_size * submatrix_size);
  std::vector<double> local_c(submatrix_size * submatrix_size, 0.0);
  std::vector<double> collected_vec(total_elements_);

  auto block_data_size = static_cast<std::size_t>(submatrix_size) * static_cast<std::size_t>(submatrix_size);

  boost::mpi::scatter(sub_world, temp_vec_1, block_1.data(), static_cast<int>(block_data_size), 0);
  boost::mpi::scatter(sub_world, temp_vec_2, block_2.data(), static_cast<int>(block_data_size), 0);

  int row = rank / block_size;
  int col = rank % block_size;

  int send_vec_1_rank = (row * block_size) + ((col + block_size - 1) % block_size);
  int recv_vec_1_rank = (row * block_size) + ((col + 1) % block_size);

  if (send_vec_1_rank >= size || recv_vec_1_rank >= size) {
    return false;
  }

  int send_vec_2_rank = col + (block_size * ((row + block_size - 1) % block_size));
  int recv_vec_2_rank = col + (block_size * ((row + 1) % block_size));

  if (send_vec_2_rank >= size || recv_vec_2_rank >= size) {
    return false;
  }

  for (int i = 0; i < row; ++i) {
    boost::mpi::request send_req;
    boost::mpi::request recv_req;

    std::vector<double> buffer_1(block_1.size());
    send_req = sub_world.isend(send_vec_1_rank, 0, block_1.data(), static_cast<int>(block_data_size));
    recv_req = sub_world.irecv(recv_vec_1_rank, 0, buffer_1.data(), static_cast<int>(block_data_size));

    if (send_req.active() && recv_req.active()) {
      send_req.wait();
      recv_req.wait();
    } else {
      return false;
    }

    block_1 = buffer_1;
  }

  for (int i = 0; i < col; ++i) {
    boost::mpi::request send_req_2;
    boost::mpi::request recv_req_2;

    std::vector<double> buffer_2(block_2.size());
    send_req_2 = sub_world.isend(send_vec_2_rank, 1, block_2.data(), static_cast<int>(block_data_size));
    recv_req_2 = sub_world.irecv(recv_vec_2_rank, 1, buffer_2.data(), static_cast<int>(block_data_size));

    if (send_req_2.active() && recv_req_2.active()) {
      send_req_2.wait();
      recv_req_2.wait();
    } else {
      return false;
    }

    block_2 = buffer_2;
  }

  for (int i = 0; i < submatrix_size; ++i) {
    for (int j = 0; j < submatrix_size; ++j) {
      for (int k = 0; k < submatrix_size; ++k) {
        local_c[(i * submatrix_size) + j] += block_1[(i * submatrix_size) + k] * block_2[(k * submatrix_size) + j];
      }
    }
  }

  for (int iter = 0; iter < block_size - 1; ++iter) {
    boost::mpi::request send_req_1;
    boost::mpi::request recv_req_1;

    boost::mpi::request send_req_2;
    boost::mpi::request recv_req_2;

    std::vector<double> buffer_1(block_1.size());
    send_req_1 = sub_world.isend(send_vec_1_rank, 0, block_1.data(), static_cast<int>(block_data_size));
    recv_req_1 = sub_world.irecv(recv_vec_1_rank, 0, buffer_1.data(), static_cast<int>(block_data_size));

    std::vector<double> buffer_2(block_2.size());
    send_req_2 = sub_world.isend(send_vec_2_rank, 1, block_2.data(), static_cast<int>(block_data_size));
    recv_req_2 = sub_world.irecv(recv_vec_2_rank, 1, buffer_2.data(), static_cast<int>(block_data_size));

    if (send_req_1.active() && recv_req_1.active() && send_req_2.active() && recv_req_2.active()) {
      send_req_1.wait();
      recv_req_1.wait();
      send_req_2.wait();
      recv_req_2.wait();
    } else {
      return false;
    }

    block_1 = buffer_1;
    block_2 = buffer_2;

    for (int i = 0; i < submatrix_size; ++i) {
      for (int j = 0; j < submatrix_size; ++j) {
        for (int k = 0; k < submatrix_size; ++k) {
          local_c[(i * submatrix_size) + j] += block_1[(i * submatrix_size) + k] * block_2[(k * submatrix_size) + j];
        }
      }
    }
  }

  auto local_c_size = local_c.size();
  boost::mpi::gather(sub_world, local_c.data(), static_cast<int>(local_c_size), collected_vec, 0);

  if (rank == 0) {
    for (int block_row = 0; block_row < block_size; ++block_row) {
      for (int block_col = 0; block_col < block_size; ++block_col) {
        int block_rank = (block_row * block_size) + block_col;
        int block_index = block_rank * submatrix_size * submatrix_size;

        for (int i = 0; i < submatrix_size; ++i) {
          for (int j = 0; j < submatrix_size; ++j) {
            int global_row = (block_row * submatrix_size) + i;
            int global_col = (block_col * submatrix_size) + j;
            result_matrix_[(global_row * matrix_size_) + global_col] =
                collected_vec[block_index + (i * submatrix_size) + j];
          }
        }
      }
    }
  }
  return true;
}
// NOLINTEND

bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output = reinterpret_cast<std::vector<double>*>(task_data->outputs[0]);
    output->assign(result_matrix_.begin(), result_matrix_.end());
  }
  return true;
}