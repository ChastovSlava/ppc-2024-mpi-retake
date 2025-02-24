// Copyright 2023 Nesterov Alexander
#include "mpi/chastov_v_algorithm_cannon/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <vector>

namespace {
int find_compatible_q(int size, int N) {
  int q = std::floor(std::sqrt(size));
  while (q > 0) {
    if (N % q == 0) {
      break;
    }
    --q;
  }
  return q > 0 ? q : 1;
}

void extract_block(const std::vector<double>& matrix, double* block, int N, int K, int block_row, int block_col) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      block[i * K + j] = matrix[(block_row * K + i) * N + (block_col * K + j)];
    }
  }
}

void multiply_blocks(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int K) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * K + j] += A[i * K + k] * B[k * K + j];
      }
    }
  }
}

void rearrange_matrix(const std::vector<double>& gathered_blocks, std::vector<double>& final_matrix, int N, int K,
                      int q) {
  for (int block_row = 0; block_row < q; ++block_row) {
    for (int block_col = 0; block_col < q; ++block_col) {
      int block_rank = block_row * q + block_col;
      int block_index = block_rank * K * K;

      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
          int global_row = block_row * K + i;
          int global_col = block_col * K + j;
          final_matrix[global_row * N + global_col] = gathered_blocks[block_index + i * K + j];
        }
      }
    }
  }
}
}  // namespace

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

bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, total_elements_, 0);

  int q = find_compatible_q(size, matrix_size_);
  int K = matrix_size_ / q;

  int color = (rank < q * q) ? 1 : MPI_UNDEFINED;

  MPI_Comm new_comm;
  MPI_Comm_split(world_, color, rank, &new_comm);

  if (color == MPI_UNDEFINED) {
    return true;
  }

  boost::mpi::communicator my_world(new_comm, boost::mpi::comm_take_ownership);
  rank = my_world.rank();
  size = my_world.size();

  std::vector<double> scatter_A(total_elements_);
  std::vector<double> scatter_B(total_elements_);
  if (rank == 0) {
    int index = 0;
    for (int block_row = 0; block_row < q; ++block_row) {
      for (int block_col = 0; block_col < q; ++block_col) {
        extract_block(first_matrix_, scatter_A.data() + index, matrix_size_, K, block_row, block_col);
        extract_block(second_matrix_, scatter_B.data() + index, matrix_size_, K, block_row, block_col);
        index += K * K;
      }
    }
  }

  std::vector<double> local_A(K * K);
  std::vector<double> local_B(K * K);
  std::vector<double> local_C(K * K, 0.0);
  std::vector<double> unfinished_C(total_elements_);

  boost::mpi::scatter(my_world, scatter_A, local_A.data(), K * K, 0);
  boost::mpi::scatter(my_world, scatter_B, local_B.data(), K * K, 0);

  int row = rank / q;
  int col = rank % q;

  int send_rank_A = row * q + (col + q - 1) % q;
  int recv_rank_A = row * q + (col + 1) % q;

  if (send_rank_A >= size || recv_rank_A >= size) {
    std::cerr << "Invalid rank for send or receive: send_rank=" << send_rank_A << ", recv_rank=" << recv_rank_A
              << std::endl;
    return false;
  }

  int send_rank_B = col + q * ((row + q - 1) % q);
  int recv_rank_B = col + q * ((row + 1) % q);

  if (send_rank_B >= size || recv_rank_B >= size) {
    std::cerr << "Invalid rank for send or receive: send_rank=" << send_rank_B << ", recv_rank=" << recv_rank_B
              << std::endl;
    return false;
  }

  for (int i = 0; i < row; ++i) {
    boost::mpi::request send_request;
    boost::mpi::request recv_request;

    std::vector<double> temp(local_A.size());
    send_request = my_world.isend(send_rank_A, 0, local_A.data(), local_A.size());
    recv_request = my_world.irecv(recv_rank_A, 0, temp.data(), temp.size());

    if (send_request.active() && recv_request.active()) {
      send_request.wait();
      recv_request.wait();
    } else {
      return false;
    }

    local_A = temp;
  }

  for (int i = 0; i < col; ++i) {
    boost::mpi::request send_request1;
    boost::mpi::request recv_request1;

    std::vector<double> temp_B(local_B.size());
    send_request1 = my_world.isend(send_rank_B, 1, local_B.data(), local_B.size());
    recv_request1 = my_world.irecv(recv_rank_B, 1, temp_B.data(), temp_B.size());

    if (send_request1.active() && recv_request1.active()) {
      send_request1.wait();
      recv_request1.wait();
    } else {
      return false;
    }

    local_B = temp_B;
  }

  multiply_blocks(local_A, local_B, local_C, K);

  for (int iter = 0; iter < q - 1; ++iter) {
    boost::mpi::request send_request2;
    boost::mpi::request recv_request2;

    boost::mpi::request send_request3;
    boost::mpi::request recv_request3;

    std::vector<double> temp_A(local_A.size());
    send_request2 = my_world.isend(send_rank_A, 0, local_A.data(), local_A.size());
    recv_request2 = my_world.irecv(recv_rank_A, 0, temp_A.data(), temp_A.size());

    std::vector<double> temp_B(local_B.size());
    send_request3 = my_world.isend(send_rank_B, 1, local_B.data(), local_B.size());
    recv_request3 = my_world.irecv(recv_rank_B, 1, temp_B.data(), temp_B.size());

    if (send_request2.active() && recv_request2.active() && send_request3.active() && recv_request3.active()) {
      send_request2.wait();
      recv_request2.wait();
      send_request3.wait();
      recv_request3.wait();
    } else {
      return false;
    }

    local_A = temp_A;
    local_B = temp_B;

    multiply_blocks(local_A, local_B, local_C, K);
  }

  boost::mpi::gather(my_world, local_C.data(), local_C.size(), unfinished_C, 0);
  if (rank == 0) {
    rearrange_matrix(unfinished_C, result_matrix_, matrix_size_, K, q);
  }
  return true;
}

/*bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, total_elements_, 0);

  int block_size = std::floor(std::sqrt(size));
  while (block_size > 0) {
    if (matrix_size_ % block_size == 0) {
      break;
    }
    --block_size;
  }
  block_size = std::max(block_size, 1);
  int submatrix_size = static_cast<int>(matrix_size_ / block_size);

  int group_color = 0;
  if (rank < block_size * block_size) {
    group_color = 1;
  } else {
    group_color = MPI_UNDEFINED;
  }

  MPI_Comm sub_comm = MPI_COMM_NULL;
  MPI_Comm_split(world_, group_color, rank, &sub_comm);

  if (group_color == MPI_UNDEFINED) {
    return true;
  }

  boost::mpi::communicator sub_world(sub_comm, boost::mpi::comm_take_ownership);
  rank = sub_world.rank();
  size = sub_world.size();

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

  boost::mpi::scatter(sub_world, temp_vec_1, block_1.data(), submatrix_size * submatrix_size, 0);
  boost::mpi::scatter(sub_world, temp_vec_2, block_2.data(), submatrix_size * submatrix_size, 0);

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
    send_req = sub_world.isend(send_vec_1_rank, 0, block_1.data(), block_1.size());
    recv_req = sub_world.irecv(recv_vec_1_rank, 0, buffer_1.data(), buffer_1.size());

    // Ожидание завершения операций
    send_req.wait();
    recv_req.wait();

    block_1 = buffer_1;
  }

  for (int i = 0; i < col; ++i) {
    boost::mpi::request send_req_2;
    boost::mpi::request recv_req_2;

    std::vector<double> buffer_2(block_2.size());
    send_req_2 = sub_world.isend(send_vec_2_rank, 1, block_2.data(), block_2.size());
    recv_req_2 = sub_world.irecv(recv_vec_2_rank, 1, buffer_2.data(), buffer_2.size());

    // Ожидание завершения операций
    send_req_2.wait();
    recv_req_2.wait();

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
    send_req_1 = sub_world.isend(send_vec_1_rank, 0, block_1.data(), block_1.size());
    recv_req_1 = sub_world.irecv(recv_vec_1_rank, 0, buffer_1.data(), buffer_1.size());

    std::vector<double> buffer_2(block_2.size());
    send_req_2 = sub_world.isend(send_vec_2_rank, 1, block_2.data(), block_2.size());
    recv_req_2 = sub_world.irecv(recv_vec_2_rank, 1, buffer_2.data(), buffer_2.size());

    // Ожидание завершения операций
    send_req_1.wait();
    recv_req_1.wait();
    send_req_2.wait();
    recv_req_2.wait();

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

  boost::mpi::gather(sub_world, local_c.data(), local_c.size(), collected_vec, 0);
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
}*/

bool chastov_v_algorithm_cannon_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output = reinterpret_cast<std::vector<double>*>(task_data->outputs[0]);
    output->assign(result_matrix_.begin(), result_matrix_.end());
  }
  return true;
}