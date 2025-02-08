// MPI tutorials: https://mpitutorial.com/tutorials/
// Compile: mpicc <source_name> -o <exe_name>
// Run: mpirun -n <num_of_programs> <exe_name>
// MPI Version: mpiexec (OpenRTE) 4.1.2

// The functions below are both BLOCKING one
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

int My_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    int rank_, size_; //, num_recv;
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int delta = rank_ ^ root;
    MPI_Status status;

    for (int i = 0; size_ > 1; size_ >>= 1, i++)
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
            // // for incorrect read count
            // MPI_Get_count(&status, MPI_INT, &num_recv);
            // if (num_recv != count)
            //     return MPI_ERR_COUNT;
        }
    
    return MPI_SUCCESS;
}

void arr_add(int *a, int *b, int n){
    // the result is in-place for a[]
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}

int My_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
    // ignore op, we only do MPI_SUM
    int rank_, size_;
    int node_tmp[MAX_LENGTH], local_recvbuf[MAX_LENGTH];
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    int delta = rank_ ^ root;
    MPI_Status status;
    
    // move the initial data into local arr
    memcpy(node_tmp, sendbuf, count * sizeof(int));

    for (int i = 0; size_ > 1; size_ >>= 1, i++)
        if ((delta & ((1 << (i + 1)) - 1)) == 0){
            // recver on round i
            // printf("Recver %d on round %d, from node %d\n", rank_, i, rank_ ^ (1 << i));
            MPI_Recv(
                local_recvbuf, count, datatype, rank_ ^ (1 << i), 0, comm,
                &status
            );
            arr_add(node_tmp, local_recvbuf, count);
        }
        else if ((delta & ((1 << i) - 1)) == 0){
            // sender on round i
            // printf("Sender %d on round %d, to node %d\n", rank_, i, rank_ ^ (1 << i));
            MPI_Send(
                node_tmp, count, datatype, rank_ ^ (1 << i), 0, comm
            );
        }
    
    // copy the result into the recvbuf
    if (delta == 0)
        memcpy(recvbuf, node_tmp, count * sizeof(int));

    return MPI_SUCCESS;
}
