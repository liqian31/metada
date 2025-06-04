#include "parallel_adapter.hpp"
#include <stdexcept>
#include <algorithm>

namespace metada {
namespace utils {

ParallelAdapter::ParallelAdapter()
    : world_comm_(MPI_COMM_WORLD)
    , model_comm_(MPI_COMM_NULL)
    , filter_comm_(MPI_COMM_NULL)
    , couple_comm_(MPI_COMM_NULL)
    , world_rank_(0)
    , world_size_(0)
    , model_rank_(0)
    , model_size_(0)
    , filter_rank_(0)
    , filter_size_(0)
    , couple_rank_(0)
    , couple_size_(0)
    , task_id_(0)
    , local_model_size_(0)
{
}

ParallelAdapter::~ParallelAdapter() {
    if (model_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&model_comm_);
    }
    if (filter_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&filter_comm_);
    }
    if (couple_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&couple_comm_);
    }
}

void ParallelAdapter::initialize(int argc, char** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }

    world_comm_ = MPI_COMM_WORLD;
    MPI_Comm_rank(world_comm_, &world_rank_);
    MPI_Comm_size(world_comm_, &world_size_);
}

void ParallelAdapter::distributeTasks(int num_model_tasks) {
    if (num_model_tasks <= 0 || num_model_tasks > world_size_) {
        throw std::runtime_error("Invalid number of model tasks");
    }

    // Calculate local model sizes
    local_model_sizes_.resize(num_model_tasks);
    int base_size = world_size_ / num_model_tasks;
    int remainder = world_size_ % num_model_tasks;

    for (int i = 0; i < num_model_tasks; ++i) {
        local_model_sizes_[i] = base_size + (i < remainder ? 1 : 0);
    }

    // Determine task_id for this process
    int offset = 0;
    for (int i = 0; i < num_model_tasks; ++i) {
        if (world_rank_ < offset + local_model_sizes_[i]) {
            task_id_ = i;
            local_model_size_ = local_model_sizes_[i];
            break;
        }
        offset += local_model_sizes_[i];
    }

    // Create communicators
    createModelComm();
    createFilterComm();
    createCoupleComm();
}

void ParallelAdapter::createModelComm() {
    // Create model communicator for each task
    int color = task_id_;
    MPI_Comm_split(world_comm_, color, world_rank_, &model_comm_);
    MPI_Comm_rank(model_comm_, &model_rank_);
    MPI_Comm_size(model_comm_, &model_size_);
}

void ParallelAdapter::createFilterComm() {
    // Only processes from task 0 join the filter communicator
    int color = (task_id_ == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world_comm_, color, world_rank_, &filter_comm_);
    
    if (filter_comm_ != MPI_COMM_NULL) {
        MPI_Comm_rank(filter_comm_, &filter_rank_);
        MPI_Comm_size(filter_comm_, &filter_size_);
    }
}

void ParallelAdapter::createCoupleComm() {
    // Create coupling communicator for each model task
    int color = task_id_;
    MPI_Comm_split(world_comm_, color, world_rank_, &couple_comm_);
    MPI_Comm_rank(couple_comm_, &couple_rank_);
    MPI_Comm_size(couple_comm_, &couple_size_);
}

void ParallelAdapter::finalize() {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

} // namespace utils
} // namespace metada 