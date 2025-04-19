// See here (https://rookiehpc.org/mpi/docs/index.html) for a better understanding about each function below
// MACROs
#define MAX_LENGTH (4096 + 0xff)
#define BROADCAST  2 // \in{1, 2}
#define REDUCE     2 // \in{1, 2}
#define ALLREDUCE  2 // \in{1, 2}
#define SCAN       2 // \in{1, 2}

/*  Part 1: Basic (each one has a root) */
// Broadcast
int My_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
// Reduce
int My_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
    MPI_Op op, int root, MPI_Comm comm);

// Scatter
int My_MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
// Gather
int My_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

/*  Part 2: Advanced (no root here) */
// All Reduce (All-Reduce) = Reduce + Broadcast
int My_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
// Scan (Prefix-Sum)
int My_MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
    MPI_Op op, MPI_Comm comm);

// All Gather (All-to-All Broadcast) = Gather + (Vectorized) Broadcast
int My_MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
// Reduce Scatter (All-to-All Reduction) = (Vectorized) Reduce + Scatter
int My_MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

// All-to-All (All-to-All Personalized Communication / Matrix Transpose (if recvcount == sendcount == 1) )
int My_MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
