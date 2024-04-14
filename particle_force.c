#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define sgn(x) (((x) < 0.0) ? (-1.0) : (1.0))
#define c1 1.23456
#define c2 6.54321

void calc_force(int n, double *x, double *f, int rank, int size) {
    int i, j, start, end;
    double diff, tmp, *buf;

    // Allocate buffer for local forces
    buf = (double *)malloc(n * sizeof(double));

    // Calculate start and end indices for this process
    start = rank * n / size;
    end = (rank + 1) * n / size;

    // Calculate local forces
    for (i = start; i < end; i++) {
        f[i] = 0.0;
        for (j = 0; j < n; j++) {
            if (j != i) {
                diff = x[i] - x[j];
                tmp = c1 / (diff * diff * diff) - c2 * sgn(diff) / (diff * diff);
                f[i] += tmp;
            }
        }
        buf[i] = f[i]; // Store local forces in buffer
    }

    // Reduce forces across processes using binary tree algorithm
    for (i = 1; i < size; i *= 2) {
        if (rank % (2 * i) != 0) {
            MPI_Send(buf, n, MPI_DOUBLE, rank - i, 0, MPI_COMM_WORLD);
            break;
        }
        if (rank + i < size) {
            double *temp = (double *)malloc(n * sizeof(double));
            MPI_Recv(temp, n, MPI_DOUBLE, rank + i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (j = start; j < end; j++) {
                f[j] += temp[j];
            }
            free(temp);
        }
    }

    // Free buffer
    free(buf);
}

int main(int argc, char *argv[]) {
    int n = 10000; // Number of particles
    int rank, size;
    double *x, *f;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory for particle positions and forces
    x = (double *)malloc(n * sizeof(double));
    f = (double *)malloc(n * sizeof(double));

    // Initialize particle positions (assuming some initialization logic)
    for (i = 0; i < n; i++) {
        x[i] = i; // Example: Particle positions initialized from 0 to n-1
    }

    // Calculate forces in parallel
    calc_force(n, x, f, rank, size);

    // Print forces from process 0
    if (rank == 0) {
        printf("Forces:\n");
        for (i = 0; i < n; i++) {
            printf("%f ", f[i]);
        }
        printf("\n");
    }

    // Free memory
    free(x);
    free(f);

    MPI_Finalize();
    return 0;
}



// Compile the MPI code
// mpicc -o mpi_force_calc mpi_force_calc.c

// Run the MPI executable with 4 processes
// mpiexec -n 4 ./mpi_force_calc
