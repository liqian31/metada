#pragma once

#include <mpi.h>
#include <memory>
#include <vector>
#include <string>

namespace metada {
namespace utils {

class ParallelAdapter {
public:
    ParallelAdapter();
    ~ParallelAdapter();

    // Initialize MPI and create communicators
    void initialize(int argc, char** argv);
    
    // Get communicator information
    MPI_Comm getWorldComm() const { return world_comm_; }
    MPI_Comm getModelComm() const { return model_comm_; }
    MPI_Comm getFilterComm() const { return filter_comm_; }
    MPI_Comm getCoupleComm() const { return couple_comm_; }

    // Get rank and size information
    int getWorldRank() const { return world_rank_; }
    int getWorldSize() const { return world_size_; }
    int getModelRank() const { return model_rank_; }
    int getModelSize() const { return model_size_; }
    int getFilterRank() const { return filter_rank_; }
    int getFilterSize() const { return filter_size_; }
    int getCoupleRank() const { return couple_rank_; }
    int getCoupleSize() const { return couple_size_; }

    // Task distribution
    void distributeTasks(int num_model_tasks);
    int getTaskId() const { return task_id_; }
    int getLocalModelSize() const { return local_model_size_; }

    // Finalize MPI
    void finalize();

private:
    // MPI communicators
    MPI_Comm world_comm_;
    MPI_Comm model_comm_;
    MPI_Comm filter_comm_;
    MPI_Comm couple_comm_;

    // Rank and size information
    int world_rank_;
    int world_size_;
    int model_rank_;
    int model_size_;
    int filter_rank_;
    int filter_size_;
    int couple_rank_;
    int couple_size_;

    // Task information
    int task_id_;
    int local_model_size_;
    std::vector<int> local_model_sizes_;

    // Helper functions
    void createModelComm();
    void createFilterComm();
    void createCoupleComm();
};

} // namespace utils
} // namespace metada 