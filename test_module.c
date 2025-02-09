#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include "HyperCubeLib.h"

// global vars
int world_rank, world_size;

void rand_init_array(int *a, int n){
    for (int i = 0; i < n; i++)
        a[i] = rand();
}

void rand_init_matrix(int **a, int n, int m){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            a[i][j] = rand();
}

bool cmp_array(int *a, int *b, int n){
    for (int i = 0; i < n; i++)
        if (a[i] != b[i])
            return 1;
    return 0;
}

bool cmp_matrix(int **a, int **b, int n, int m){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (a[i][j] != b[i][j])
                return 1;
    return 0;   
}

void unified_root_count(int *count_pt, int *root_pt){
    int tmp[2] = {0};
    if (world_rank == 0){
        tmp[0] = (int)((rand() / (float)RAND_MAX) * (MAX_LENGTH - 0xff) + 0x3f) & (~(world_size - 1));
        tmp[1] = (rand() / (float)RAND_MAX) * (world_size - 1);
    }
    MPI_Bcast(tmp, 2, MPI_INT, 0, MPI_COMM_WORLD);
    *count_pt = tmp[0];
    *root_pt = tmp[1];
}

void exit_code(char *out_code, int root, int count){
    printf("%s Testing Failed on node %d (world_size = %d, and (root, count) = %d, %d)\n", 
        out_code, world_rank, world_size, root, count);
    MPI_Finalize();
    exit(0);
}   

void Bcast_test(){
    int count, root;
    int arr_std[MAX_LENGTH], arr_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);

    // begin to test
    if (world_rank == root){
        printf("----------- Broadcast Testing -------------\n");
        rand_init_array(arr_std, count);
        memcpy(arr_my, arr_std, count * sizeof(int));
    }
    MPI_Bcast(arr_std, count, MPI_INT, root, MPI_COMM_WORLD);
    My_MPI_Bcast(arr_my, count, MPI_INT, root, MPI_COMM_WORLD);
    if (cmp_array(arr_my, arr_std, count))
        exit_code("Broadcast", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

void Reduce_test(){
    int count, root;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    rand_init_array(send_std, count);
    memcpy(send_my, send_std, count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- Reduce Testing -------------\n");
    MPI_Reduce(send_std, recv_std, count, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    My_MPI_Reduce(send_my, recv_my, count, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    if (root == world_rank && cmp_array(recv_std, recv_my, count))
        exit_code("Reduce", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

void Scatter_test(){
    int count, root, sdecv_count;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    sdecv_count = count / world_size;

    // begin to test
    if (world_rank == root){
        printf("----------- Scatter Testing -------------\n");
        rand_init_array(send_std, count);
        memcpy(send_my, send_std, count * sizeof(int));
    }
    MPI_Scatter(send_std, sdecv_count, MPI_INT, recv_std, sdecv_count, MPI_INT, root, MPI_COMM_WORLD);
    My_MPI_Scatter(send_my, sdecv_count, MPI_INT, recv_my, sdecv_count, MPI_INT, root, MPI_COMM_WORLD);
    if (cmp_array(recv_my, recv_std, sdecv_count))
        exit_code("Scatter", root, count);

    MPI_Barrier(MPI_COMM_WORLD);
}

void Gather_test(){
    int count, root, sdecv_count;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    sdecv_count = count / world_size;
    rand_init_array(send_std, sdecv_count);
    memcpy(send_my, send_std, sdecv_count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- Gather Testing -------------\n");
    MPI_Gather(send_std, sdecv_count, MPI_INT, recv_std, sdecv_count, MPI_INT, root, MPI_COMM_WORLD);
    My_MPI_Gather(send_my, sdecv_count, MPI_INT, recv_my, sdecv_count, MPI_INT, root, MPI_COMM_WORLD);
    if (root == world_rank && cmp_array(recv_my, recv_std, count))
        exit_code("Gather", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

// now, the root in below functions is just for printf usage...

void AllReduce_test(){
    int count, root;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    rand_init_array(send_std, count);
    memcpy(send_my, send_std, count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- All Reduce Testing -------------\n");
    MPI_Allreduce(send_std, recv_std, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    My_MPI_Allreduce(send_my, recv_my, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (cmp_array(recv_std, recv_my, count))
        exit_code("All Reduce", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

void Scan_test(){
    int count, root;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    rand_init_array(send_std, count);
    memcpy(send_my, send_std, count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- Scan Testing -------------\n");
    MPI_Scan(send_std, recv_std, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    My_MPI_Scan(send_my, recv_my, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (cmp_array(recv_std, recv_my, count))
        exit_code("Scan", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

void AllGather_test(){
    int count, root, sdecv_count;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    sdecv_count = count / world_size;
    rand_init_array(send_std, sdecv_count);
    memcpy(send_my, send_std, sdecv_count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- All Gather Testing -------------\n");
    MPI_Allgather(send_std, sdecv_count, MPI_INT, recv_std, sdecv_count, MPI_INT, MPI_COMM_WORLD);
    My_MPI_Allgather(send_my, sdecv_count, MPI_INT, recv_my, sdecv_count, MPI_INT, MPI_COMM_WORLD);
    if (cmp_array(recv_my, recv_std, count))
        exit_code("All Gather", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

void ReduceScatter_test(){
    int count, root, sdecv_count[world_size + 3];
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    for (int i = 0; i < world_size; i++)
        sdecv_count[i] = count / world_size;
    rand_init_array(send_std, count);
    memcpy(send_my, send_std, count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- Reduce Scatter Testing -------------\n");
    MPI_Reduce_scatter(send_std, recv_std, sdecv_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    My_MPI_Reduce_scatter(send_my, recv_my, sdecv_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (cmp_array(recv_my, recv_std, sdecv_count[world_rank]))
        exit_code("Reduce Scatter", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

void AllToAll_test(){
    int count, root, sdecv_count;
    int send_std[MAX_LENGTH], send_my[MAX_LENGTH];
    int recv_std[MAX_LENGTH], recv_my[MAX_LENGTH];

    // initialization
    unified_root_count(&count, &root);
    sdecv_count = count / world_size;
    rand_init_array(send_std, count);
    memcpy(send_my, send_std, count * sizeof(int));

    // begin to test
    if (world_rank == root)
        printf("----------- All To All Testing -------------\n");
    MPI_Alltoall(send_std, sdecv_count, MPI_INT, recv_std, sdecv_count, MPI_INT, MPI_COMM_WORLD);
    My_MPI_Alltoall(send_my, sdecv_count, MPI_INT, recv_my, sdecv_count, MPI_INT, MPI_COMM_WORLD);
    if (cmp_array(recv_my, recv_std, count))
        exit_code("All To All", root, count);
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv); 
    srand(time(NULL));
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // test broadcast
    Bcast_test();
    // test Reduce
    Reduce_test();
    // test Scatter
    Scatter_test();
    // test Gather
    Gather_test();

    // test AllReduce
    AllReduce_test();
    // test Scan
    Scan_test();
    // test All Gather
    AllGather_test();
    // test Reduce Scatter
    ReduceScatter_test();
    // test ALL to ALL
    AllToAll_test();

    if (world_rank == 0)
        printf("-------- ALL TEST PASSED! -----------\n");
    MPI_Finalize();
    return 0;
}