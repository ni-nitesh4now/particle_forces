#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define c1 1.23456
#define c2 6.54321
#define n 840000

// Function to calculate forces for a subset of particles
void calcForce(int local_n, double *local_x, double *local_f) {
    int i, j;
    double diff, temp, tmp;

    // Initialize local forces to zero
    for (i = 0; i < local_n; i++)
        local_f[i] = 0.0;

    // Calculate forces for local subset of particles
    for (i = 1; i < local_n; i++) {
        for (j = 0; j < i; j++) {
            diff = local_x[i] - local_x[j];
            temp = 1.0 / (diff * diff * diff);
            tmp = c1 * temp * temp - c2 * temp;
            local_f[i] += tmp;
            local_f[j] -= tmp;
        }
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

    // Initialize particle positions (assuming x is initialized appropriately)
    // Example: Initialize x array (each processor has its own local_x)
    for (int i = start_idx; i < end_idx; i++) {
        x[i] = i; // Example: Initialize particle positions (e.g., linear distribution)
    }

    // Scatter particle positions to all processors
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local forces
    calcForce(local_n, local_x, local_f);

    // Gather local forces to the root processor (rank 0)
    MPI_Gather(local_f, local_n, MPI_DOUBLE, f, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform reduction to aggregate forces across all processors
    MPI_Allreduce(MPI_IN_PLACE, f, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Print the forces on process 0
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

// script my_terminal_session.txt
// mpicc -o force_calculation_mpi force_calculation_mpi.c
// mpirun -np 4 ./force_calculation_mpi
