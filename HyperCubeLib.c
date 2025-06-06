// MPI tutorials: https://mpitutorial.com/tutorials/
// Compile: mpicc <source_name> -o <exe_name>
// Run: mpirun -n <num_of_programs> <exe_name>
// MPI Version: mpiexec (OpenRTE) 4.1.2

// MPI_Send(
//     void* data,
//     int count,
//     MPI_Datatype datatype,
//     int destination,
//     int tag,
//     MPI_Comm communicator)
// MPI_Recv(
//     void* data,
//     int count,
//     MPI_Datatype datatype,
//     int source,
//     int tag,
//     MPI_Comm communicator,
//     MPI_Status* status)
// MPI_Barrier(
//     MPI_Comm communicator)

#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "HyperCubeLib.h"

// See here (https://rookiehpc.org/mpi/docs/index.html) for a better understanding about:
//     1. what is each function doing
//     2. what is the meaning of each argument
// for simplification, the implementation will not check for the error code
// for simplification, let's assume world_size = 2^t
// for simplification, for reduce we only do MPI_SUM
// for simplification, we do not implement MPI_Gatherv and MPI_Scatterv, 
//      and assume sendcount = recvcount for MPI_Scatter / MPI_Gather / MPI_Allgather
//      and assume recvcounts[] is a constant array for MPI_Reduce_scatter

void arr_add(int *a, int *b, int n){
    // the result is in-place for a[]
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}

void arr_mul_add_mod(int *a, int *b, int *c, int n){
    // the result is in-place for a[]
    for (int i = 0; i < n; i++)
        a[i] = (a[i] + b[i] * (ll)c[i]) % my_mod;
}

void arr_mul_mod(int *a, int *b, int n){
    // the result is in-place for a[]
    for (int i = 0; i < n; i++)
        a[i] = (a[i] * (ll)b[i]) % my_mod;
}

/*  Part 1: Basic (each one has a root) */

int Bcast_impl_1(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m log n
        work:           n
        msg work:       nm
    */
    int rank_, size_;
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int delta = rank_ ^ root;
    MPI_Status status;

    /*
        msg mode: (delta)
        assume size_ = 2^k
        round 0: 0 -> 1                            (2 nodes involved)
        round 1: 0 -> 2, 1 -> 3                    (4 nodes involved)
        round 2: 0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7    (8 nodes involved)
        ...
        (last)                                     (2^k nodes involved)
    */
    for (int i = 0; (1 << i) < size_; i++)
        if (delta < (1 << i)){
            // sender on round i
            MPI_Send(
                buffer, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
        else if (delta < (1 << (i + 1))){
            // recver on round i
            MPI_Recv(
                buffer, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
        }
    
    return MPI_SUCCESS;
}

int Bcast_impl_2(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          2 * log n
        msg depth:      2 * m
        work:           n + n log n
        msg work:       m (n + log n)
    */
    int size_;
    MPI_Comm_size(comm, &size_);
    int sdecv_count = count / size_;
    int buff_[sdecv_count + 5];

    My_MPI_Scatter(buffer, sdecv_count, MPI_INT, buff_, sdecv_count, MPI_INT, root, comm);
    My_MPI_Allgather(buff_, sdecv_count, MPI_INT, buffer, sdecv_count, MPI_INT, comm);

    return MPI_SUCCESS;
}

int My_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    if (BROADCAST == 1)
        return Bcast_impl_1(buffer, count, datatype, root, comm);
    else if (BROADCAST == 2)
        return Bcast_impl_2(buffer, count, datatype, root, comm);
    else{
        printf("Not Implemented Error from Broadcast!\n");
        exit(1);
    }
}

int Reduce_impl_1(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m log n
        work:           n
        msg work:       nm
    */
    // ignore op, we only do MPI_SUM
    int rank_, size_;
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int delta = rank_ ^ root;
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * sizeof(int));

    /*
        msg mode: (delta)
        assume size_ = 2^k
        round 0: 0 <- 1, 2 <- 3, 4 <- 5, ...       (2^k nodes involved)
        round 1: 0 <- 2, 4 <- 6, 8 <- 10, ...      (2^{k-1} nodes involved)
        round 2: 0 <- 4, 8 <- 12, 16 <- 20, ...    (2^{k-2} nodes involved)
        ...
        (last)                                     (2 nodes involved)
    */
    for (int i = 0; (1 << i) < size_; i++)
        if ((delta & ((1 << (i + 1)) - 1)) == 0){
            // recver on round i
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            arr_add(node_tmp, local_recvbuf, count);
        }
        else if ((delta & ((1 << i) - 1)) == 0){
            // sender on round i
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
    
    // copy the result into the recvbuf
    if (delta == 0)
        memcpy(recvbuf, node_tmp, count * sizeof(int));

    return MPI_SUCCESS;
}

int Reduce_impl_2(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          2 * log n
        msg depth:      2 * m
        work:           n + n log n
        msg work:       m (n + log n)
    */
    int size_;
    MPI_Comm_size(comm, &size_);
    int sdecv_count = count / size_;
    int buff_[sdecv_count + 5];

    My_MPI_Reduce_scatter(sendbuf, buff_, &sdecv_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    My_MPI_Gather(buff_, sdecv_count, MPI_INT, recvbuf, sdecv_count, MPI_INT, root, MPI_COMM_WORLD);

    return MPI_SUCCESS;
}

int My_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
    if (REDUCE == 1)
        return Reduce_impl_1(sendbuf, recvbuf, count, datatype, op, root, comm);
    else if (REDUCE == 2)
        return Reduce_impl_2(sendbuf, recvbuf, count, datatype, op, root, comm);
    else{
        printf("Not Implemented Error from Reduce!\n");
        exit(1);
    }
}

int My_MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m         ..... {round 0: m/2, round 1: m/4, ...}
        work:           n
        msg work:       m log n   ..... {each layer has m in total, and log n layers in total}
    */
    int rank_, size_;
    int local_buf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int delta = rank_ ^ root, i = 0;
    int count = recvcount * size_; // the length of total msg
    MPI_Status status;

    // move the initial data into local arr
    if (rank_ == root)
        memcpy(local_buf, sendbuf, sendcount * size_ * sizeof(int));

    /*
        msg mode: (delta)
        assume size_ = 2^k
        round 0: 0 -> 2^(k-1)                                       (2 nodes involved)
        round 1: 0 -> 2^(k-2), 2^(k-1) -> 2^(k-1) + 2^(k-2)         (4 nodes involved)
        ...
        (last)                                                      (2^k nodes involved)
    */
    while((1 << i) < size_) i++;
    for (i--; i >= 0; i--){
        count >>= 1;
        if ((delta & ((1 << (i + 1)) - 1)) == 0){
            // sender
            if (rank_ & (1 << i)){
                // send to a lower rank group
                MPI_Send(
                    local_buf, count, sendtype, rank_ ^ (1 << i), 0, comm
                );
                memcpy(local_buf, local_buf + count, count * sizeof(int));
            }
            else{
                // send to a higher rank group
                MPI_Send(
                    local_buf + count, count, sendtype, rank_ ^ (1 << i), 0, comm
                );
            }
        }
        else if ((delta & ((1 << i) - 1)) == 0){
            // recver
            MPI_Recv(
                local_buf, count, recvtype, rank_ ^ (1 << i), 0, comm,
                &status
            );
        }
    }

    // copy the result into the recvbuf
    memcpy(recvbuf, local_buf, recvcount * sizeof(int));
    
    return MPI_SUCCESS;
}

int My_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m         ..... {round 0: m/2, round 1: m/4, ...}
        work:           n
        msg work:       m log n   ..... {each layer has m in total, and log n layers in total}
    */
    int rank_, size_;
    int local_buf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int delta = rank_ ^ root;
    int count = sendcount; // the length of single msg
    MPI_Status status;

    // move the initial data into local arr
    memcpy(local_buf, sendbuf, count * sizeof(int));

    /*
        msg mode: (delta)
        assume size_ = 2^k
        round 0: 0 <- 1, 2 <- 3, 4 <- 5, ...       (2^k nodes involved)
        round 1: 0 <- 2, 4 <- 6, 8 <- 10, ...      (2^{k-1} nodes involved)
        round 2: 0 <- 4, 8 <- 12, 16 <- 20, ...    (2^{k-2} nodes involved)
        ...
        (last)                                     (2 nodes involved)
    */
    for (int i = 0; (1 << i) < size_; i++){
        if ((delta & ((1 << (i + 1)) - 1)) == 0){
            // recver on round i
            if (rank_ & (1 << i)){
                // recv from a lower rank group
                memcpy(local_buf + count, local_buf, sizeof(int) * count);
                MPI_Recv(
                    local_buf, count, recvtype, rank_ ^ (1 << i), 0, comm,
                    &status
                );
            }
            else{
                // recv from a higher rank group
                MPI_Recv(
                    local_buf + count, count, recvtype, rank_ ^ (1 << i), 0, comm,
                    &status
                );
            }
        }
        else if ((delta & ((1 << i) - 1)) == 0){
            // sender on round i
            MPI_Send(
                local_buf, count, sendtype, rank_ ^ (1 << i), 0, comm
            );
        }
        count <<= 1;
    }
    
    // copy the result into the recvbuf
    if (delta == 0)
        memcpy(recvbuf, local_buf, count * sizeof(int));

    return MPI_SUCCESS;
}

/*  Part 2: Advanced (no root here) */

int AllReduce_impl_1(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m log n
        work:           n log n
        msg work:       mn log n
    */
    // ignore op, we only do MPI_SUM
    int rank_, size_;
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 <-> 1, 2 <-> 3, 4 <-> 5, ...       (2^k nodes involved)
        round 1: 0 <-> 2, 1 <-> 3, 2 <-> 4, ...       (2^k nodes involved)
        round 2: 0 <-> 4, 1 <-> 5, 2 <-> 6, ...       (2^k nodes involved)
        ...
        (last)                                        (2^k nodes involved)
    */
    for (int i = 0; (1 << i) < size_; i++){
        if ((1 << i) & rank_){
            // right side node
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
        }
        else{
            // left side node
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
        arr_add(node_tmp, local_recvbuf, count);
    }

    
    // copy the result into the recvbuf
    memcpy(recvbuf, node_tmp, count * sizeof(int));

    return MPI_SUCCESS;
}

int AllReduce_impl_2(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          2 * log n
        msg depth:      2 * m
        work:           2 * n log n
        msg work:       2 * mn
    */
    int size_;
    MPI_Comm_size(comm, &size_);
    int sdecv_count = count / size_;
    int buff_[sdecv_count + 5];

    // All reduce = reduce + broadcast, and using reduce2 + broadcast2, 
    //      while the gather and scatter in the middle diminished
    My_MPI_Reduce_scatter(sendbuf, buff_, &sdecv_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    My_MPI_Allgather(buff_, sdecv_count, MPI_INT, recvbuf, sdecv_count, MPI_INT, comm);

    return MPI_SUCCESS;
}

int My_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    if (ALLREDUCE == 1)
        return AllReduce_impl_1(sendbuf, recvbuf, count, datatype, op, comm);
    else if (ALLREDUCE == 2)
        return AllReduce_impl_2(sendbuf, recvbuf, count, datatype, op, comm);
    else{
        printf("Not Implemented Error from All Reduce!\n");
        exit(1);
    }
}

int My_MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    if (SCAN == 1)
        return Scan_impl_1(sendbuf, recvbuf, count, datatype, op, comm);
    else if (SCAN == 2)
        return Scan_impl_2(sendbuf, recvbuf, count, datatype, op, comm);
    else{
        printf("Not Implemented Error from Scan!\n");
        exit(1);
    }
}

int Scan_impl_1(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          2 * log n
        msg depth:      2 * m log n
        work:           2 * n
        msg work:       2 * nm
    */
    // ignore op, we only do MPI_SUM
    int rank_, size_, i;
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 -> 1, 2 -> 3, 4 -> 5, ...        (2^k nodes involved)
        round 1: 1 -> 3, 5 -> 7, 9 -> 11, ...       (2^{k-1} nodes involved)
        round 2: 3 -> 7, 11 -> 15, 19 -> 23, ...    (2^{k-2} nodes involved)
        ...
        round k-1: (2^{k-1}-1) -> (2^k-1)           (2 nodes involved) 
        round k: (2^{k-1}-1) -> (2^{k-1}+2^{k-2}-1) (2^2-2 nodes involved)
        ...
        (last)                                      (2^k-2 nodes involved)
    */

    // first k-1 rounds
    for (i = 0; (1 << i) < size_; i++)
        if (((~rank_) & ((1 << (i + 1)) - 1)) == 0){
            // recv side
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            arr_add(node_tmp, local_recvbuf, count);
        }
        else if (((~rank_) & ((1 << i) - 1)) == 0){
            // send side
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
    // rest rounds
    for (i -= 2; i >= 0; i--)
        if (((~rank_) & ((1 << (i + 1)) - 1)) == 0){
            // the only skipped one which does not need to send
            if (rank_ == size_ - 1)
                continue;
            // send side
            MPI_Send(
                node_tmp, count, datatype, rank_ + (1 << i), 0, comm
            );
            
        }
        else if (((~rank_) & ((1 << i) - 1)) == 0){
            // the only skipped one which does not need to recv
            if (rank_ == ((1 << i) - 1))
                continue;
            // recv side
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ - (1 << i), 0, comm,
                &status
            );
            arr_add(node_tmp, local_recvbuf, count);
        }
    
    // copy the result into the recvbuf
    memcpy(recvbuf, node_tmp, count * sizeof(int));

    return MPI_SUCCESS;
}

int Scan_impl_2(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m log n
        work:           n log n
        msg work:       nm log n
    */
    // ignore op, we only do MPI_SUM
    int rank_, size_, i;
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    int ret_[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * sizeof(int));
    // move the initial data into the return result array
    memcpy(ret_, sendbuf, count * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 <-> 1, 2 <-> 3, 4 <-> 5, ...        (2^k nodes involved)
        round 1: 0 <-> 2, 1 <-> 3, 4 <-> 6, ...        (2^k nodes involved)
        ...
        round k-1: 0 <-> 2^{k-1}, ...                  (2^k nodes involved) 
    */

    // first k-1 rounds
    for (i = 0; (1 << i) < size_; i++){
        if ((1 << i) & rank_){
            // right side node
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            // only for the right side node, we will add the sum from another node into the prefix-sum
            arr_add(ret_, local_recvbuf, count);
        }
        else{
            // left side node
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
        arr_add(node_tmp, local_recvbuf, count);
    }
    
    // copy the result into the recvbuf
    memcpy(recvbuf, ret_, count * sizeof(int));

    return MPI_SUCCESS;
}

int My_MPI_WeightedScan(const void *sendaddbuf, const void *sendmulbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          2 * 2 * log n              <double send-recv pair, same for the rest below>
        msg depth:      2 * 2 * m log n
        work:           2 * 2 * n
        msg work:       2 * 2 * nm
    */
    // ignore op, we only do MPI_SUM
    int rank_, size_, i;
    int local_sum[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    int local_mul[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(local_sum, sendaddbuf, count * sizeof(int));
    memcpy(local_mul, sendmulbuf, count * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 -> 1, 2 -> 3, 4 -> 5, ...        (2^k nodes involved)
        round 1: 1 -> 3, 5 -> 7, 9 -> 11, ...       (2^{k-1} nodes involved)
        round 2: 3 -> 7, 11 -> 15, 19 -> 23, ...    (2^{k-2} nodes involved)
        ...
        round k-1: (2^{k-1}-1) -> (2^k-1)           (2 nodes involved) 
        round k: (2^{k-1}-1) -> (2^{k-1}+2^{k-2}-1) (2^2-2 nodes involved)
        ...
        (last)                                      (2^k-2 nodes involved)
    */

    // first k-1 rounds
    for (i = 0; (1 << i) < size_; i++)
        if (((~rank_) & ((1 << (i + 1)) - 1)) == 0){
            // first recv the sum
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            arr_mul_add_mod(
                local_sum,
                local_recvbuf,
                local_mul,
                count
            );
            // then is the mul
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            arr_mul_mod(
                local_mul,
                local_recvbuf,
                count
            );
        }
        else if (((~rank_) & ((1 << i) - 1)) == 0){
            // send side
                // first is the sum
            MPI_Send(
                local_sum, count, datatype, rank_ ^ (1 << i), 0, comm
            );
                // second is the mul
            MPI_Send(
                local_mul, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
    // rest rounds
    for (i -= 2; i >= 0; i--)
        if (((~rank_) & ((1 << (i + 1)) - 1)) == 0){
            // the only skipped one which does not need to send
            if (rank_ == size_ - 1)
                continue;
            // send side
                // first is the sum
            MPI_Send(
                local_sum, count, datatype, rank_ + (1 << i), 0, comm
            );
                // second is the mul
            MPI_Send(
                local_mul, count, datatype, rank_ + (1 << i), 0, comm
            );
        }
        else if (((~rank_) & ((1 << i) - 1)) == 0){
            // the only skipped one which does not need to recv
            if (rank_ == ((1 << i) - 1))
                continue;
            // first recv the sum
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ - (1 << i), 0, comm,
                &status
            );
            arr_mul_add_mod(
                local_sum,
                local_recvbuf,
                local_mul,
                count
            );
            // then is the mul
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ - (1 << i), 0, comm,
                &status
            );
            arr_mul_mod(
                local_mul,
                local_recvbuf,
                count
            );
        }
    
    // copy the result into the recvbuf
    memcpy(recvbuf, local_sum, count * sizeof(int));

    return MPI_SUCCESS;
}

int My_MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m
        work:           n log n
        msg work:       mn
    */
    int rank_, size_, count = sendcount;
    int local_buf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(local_buf, sendbuf, count * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 <-> 1, 2 <-> 3, 4 <-> 5, ...       (2^k nodes involved)
        round 1: 0 <-> 2, 1 <-> 3, 2 <-> 4, ...       (2^k nodes involved)
        round 2: 0 <-> 4, 1 <-> 5, 2 <-> 6, ...       (2^k nodes involved)
        ...
        (last)                                        (2^k nodes involved)
    */
    for (int i = 0; (1 << i) < size_; i++){
        if ((1 << i) & rank_){
            // right side node
            MPI_Send(
                local_buf, count, sendtype, rank_ ^ (1 << i), 0, comm
            );
            memcpy(local_buf + count, local_buf, count * sizeof(int));
            MPI_Recv(
                local_buf, count, recvtype, rank_ ^ (1 << i), 0, comm,
                &status
            );
        }
        else{
            // left side node
            MPI_Recv(
                local_buf + count, count, recvtype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            MPI_Send(
                local_buf, count, sendtype, rank_ ^ (1 << i), 0, comm
            );
        }

        count <<= 1;
    }
    
    // copy the result into the recvbuf
    memcpy(recvbuf, local_buf, count * sizeof(int));

    return MPI_SUCCESS;
}

int My_MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m
        work:           n log n
        msg work:       mn
    */
    // ignore op, we only do MPI_SUM
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    int rank_, size_;
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int count = recvcounts[0] * size_;
    MPI_Status status;

    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 <-> 2^{k-1}, 1 <-> 2^{k-1} + 1,  ...       (2^k nodes involved)
        round 1: 0 <-> 2^{k-2}, 1 <-> 2^{k-2} + 1,  ...       (2^k nodes involved)
        ...
        (last)                                                (2^k nodes involved)
    */
    int i = 0;
    while((1 << i) < size_) i++;
    for (i-- ; i >= 0; i--){
        count >>= 1;
        if ((1 << i) & rank_){
            // right side node
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
            memcpy(node_tmp, node_tmp + count, count * sizeof(int));
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
        }
        else{
            // left side node
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            MPI_Send(
                node_tmp + count, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
        arr_add(node_tmp, local_recvbuf, count);
    }
    
    // copy the result into the recvbuf
    memcpy(recvbuf, node_tmp, count * sizeof(int));

    return MPI_SUCCESS;
}

void comb_arr(int *dst, int *src, int len_, int stride){
    if (stride > 4){
        for (int i = 0; i < len_; i += stride){;
            memcpy(dst + (i << 1), src + i, stride * sizeof(int));
            memcpy(dst + (i << 1) + stride, src + i + len_, stride * sizeof(int));
        }
    }
    else if (stride == 1){
        for (int i = 0; i < len_; i++){
            dst[i << 1] = src[i];
            dst[(i << 1) | 1] = src[i + len_];
        }
    }
    else if (stride == 2){
        for (int i = 0; i < len_; i += 2){
            dst[(i << 1)] = src[i];
            dst[(i << 1) + 1] = src[i + 1];
            dst[(i << 1) + 2] = src[i + len_];
            dst[(i << 1) + 3] = src[i + len_ + 1];
        }
    }
    else if (stride == 3){
        for (int i = 0; i < len_; i += 3){
            dst[(i << 1)] = src[i];
            dst[(i << 1) + 1] = src[i + 1];
            dst[(i << 1) + 2] = src[i + 2];
            dst[(i << 1) + 3] = src[i + len_];
            dst[(i << 1) + 4] = src[i + len_ + 1];
            dst[(i << 1) + 5] = src[i + len_ + 2];
        }
    }
    else if (stride == 4){
        for (int i = 0; i < len_; i += 4){
            dst[(i << 1)] = src[i];
            dst[(i << 1) + 1] = src[i + 1];
            dst[(i << 1) + 2] = src[i + 2];
            dst[(i << 1) + 3] = src[i + 3];
            dst[(i << 1) + 4] = src[i + len_];
            dst[(i << 1) + 5] = src[i + len_ + 1];
            dst[(i << 1) + 6] = src[i + len_ + 2];
            dst[(i << 1) + 7] = src[i + len_ + 3];
        }
    }
}

int My_MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
    /*
        Assume n is the number of the nodes,
               m is the amount of total msg
        depth:          log n
        msg depth:      m log n
        work:           n log n
        msg work:       mn log n
    */
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    int rank_, size_;
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int count = (sendcount * size_) >> 1;
    MPI_Status status;

    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * 2 * sizeof(int));

    /*
        msg mode: (absolute)
        assume size_ = 2^k
        round 0: 0 <-> 2^{k-1}, 1 <-> 2^{k-1} + 1,  ...       (2^k nodes involved)
        round 1: 0 <-> 2^{k-2}, 1 <-> 2^{k-2} + 1,  ...       (2^k nodes involved)
        ...
        (last)                                                (2^k nodes involved)
    */
    int i = 0;
    while((1 << i) < size_) i++;
    for (i-- ; i >= 0; i--){
        if ((1 << i) & rank_){
            // right side node
            MPI_Send(
                node_tmp, count, sendtype, rank_ ^ (1 << i), 0, comm
            );
            memcpy(local_recvbuf + count, node_tmp + count, count * sizeof(int));
            MPI_Recv(
                local_recvbuf, count, recvtype, rank_ ^ (1 << i), 0, comm,
                &status
            );
        }
        else{
            // left side node
            MPI_Recv(
                local_recvbuf + count, count, recvtype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            memcpy(local_recvbuf, node_tmp, count * sizeof(int));
            MPI_Send(
                node_tmp + count, count, sendtype, rank_ ^ (1 << i), 0, comm
            );
        }
        comb_arr(node_tmp, local_recvbuf, count, sendcount);
    }
    
    // copy the result into the recvbuf
    memcpy(recvbuf, node_tmp, count * 2 * sizeof(int));

    return MPI_SUCCESS;
}