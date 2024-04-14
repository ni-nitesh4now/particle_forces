#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define c1 1.23456
#define c2 6.54321
#define n 840000

void calcForce(int local_n, double *local_x, double *local_f) {
    // Calculate forces for local subset of particles
    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < n; j++) {
            if (j != (i / local_n)) {  // Ensure i != j within this process
                double diff = local_x[i] - local_x[j];
                double temp = 1.0 / (diff * diff * diff);
                double tmp = c1 * temp * temp - c2 * temp;
                local_f[i] += tmp;
            }
        }
    }
}

void reduceForces(double *local_f, int local_n, int rank, int size) {
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            int partner = rank + step;
            if (partner < size) {
                double *recv_f = (double *)malloc(local_n * sizeof(double));
                MPI_Recv(recv_f, local_n, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < local_n; i++) {
                    local_f[i] += recv_f[i];
                }
                free(recv_f);
            }
        } else {
            int partner = rank - step;
            MPI_Send(local_f, local_n, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *x, *f, *local_x, *local_f;
    int local_n, start_idx, end_idx;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine local workload distribution
    local_n = n / size;
    start_idx = rank * local_n;
    end_idx = start_idx + local_n;

    // Allocate memory for arrays
    x = (double *)malloc(n * sizeof(double));
    f = (double *)malloc(n * sizeof(double));
    local_x = (double *)malloc(local_n * sizeof(double));
    local_f = (double *)calloc(local_n, sizeof(double)); // Initialize local forces to zero

    if (x == NULL || f == NULL || local_x == NULL || local_f == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize particle positions (e.g., linear distribution)
    for (int i = start_idx; i < end_idx; i++) {
        x[i] = i; // Example: Initialize particle positions (e.g., linear distribution)
    }

    // Scatter particle positions to all processes
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local forces
    calcForce(local_n, local_x, local_f);

    // Reduce local forces across all processes using binary tree algorithm
    reduceForces(local_f, local_n, rank, size);

    // Gather reduced forces to the root processor (rank 0)
    MPI_Gather(local_f, local_n, MPI_DOUBLE, f, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform reduction to aggregate forces across all processors
    MPI_Allreduce(MPI_IN_PLACE, f, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Print the forces on process 0 for debugging
    if (rank == 0) {
        printf("Forces:\n");
        for (int i = 0; i < n; i++) {
            printf("%f ", f[i]);
        }
        printf("\n");
    }

    // Clean up
    free(x);
    free(f);
    free(local_x);
    free(local_f);

    MPI_Finalize();

    return 0;
}
