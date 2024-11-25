
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>


#define MAX_ITERATIONS 50
#define VERBOSE false // show the actual matrix output & full validation matrices, disable when just benchmarking
#define VALIDATION false // validate results, disable for improved performance
#define DEBUG false


// Functions
void l_u_d_s(float **a, float **l, float **u, int n);
void l_u_d_p(float **a, float **l, float **u, int n);
void initialize_matrices(float ***a, float ***l, float ***u, int n);
void random_fill(float **matrix, int n);
void matrix_validation(float **a, float **l, float **u, int n);
void printMatrix(double **matrix, int n);
int backsubstitution_p(double *A, double *b, double *x, int n);
int backsubstitution_s(double *A, double *b, double *x, int n);
int conjugategradient_p(double *A, double *b, double *x, int n);
int conjugategradient_s(double *A, double *b, double *x, int n);
int seidel_p(double *A, double *b, double *x, int n);
int seidel_s(double *A, double *b, double *x, int n);


// Function to print a matrix
void printMatrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double* gaussJordanInverseSerial(double *matrix, int n) {
    // Create an augmented matrix
    double *augmented = (double *)malloc(n * 2 * n * sizeof(double));

    // Initialize the augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i * (2 * n) + j] = matrix[i * n + j]; // Original matrix
        }
        for (int j = n; j < 2 * n; j++) {
            augmented[i * (2 * n) + j] = (i == (j - n)) ? 1.0 : 0.0; // Identity matrix
        }
    }

    // Perform Gauss-Jordan elimination (serial)
    for (int i = 0; i < n; i++) {
        int pivot = i;

        // Find the pivot row
        for (int j = i + 1; j < n; j++) {
            if (fabs(augmented[j * (2 * n) + i]) > fabs(augmented[pivot * (2 * n) + i])) {
                pivot = j;
            }
        }

        // Swap rows if needed
        for (int k = 0; k < 2 * n; k++) {
            double temp = augmented[i * (2 * n) + k];
            augmented[i * (2 * n) + k] = augmented[pivot * (2 * n) + k];
            augmented[pivot * (2 * n) + k] = temp;
        }

        // Normalize the pivot row
        double pivotValue = augmented[i * (2 * n) + i];
        for (int j = 0; j < 2 * n; j++) {
            augmented[i * (2 * n) + j] /= pivotValue;
        }

        // Eliminate other rows
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double factor = augmented[j * (2 * n) + i];
                for (int k = 0; k < 2 * n; k++) {
                    augmented[j * (2 * n) + k] -= factor * augmented[i * (2 * n) + k];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    double *inverse = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i * n + j] = augmented[i * (2 * n) + j + n];
        }
    }

    // Free the augmented matrix
    free(augmented);

    return inverse;
}

// Function to perform Gauss-Jordan matrix inversion with single-pointer matrix initialization
double* gaussJordanInverse(double *matrix, int n) {
    // Create an augmented matrix
    int num_threads;
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);
   
     omp_set_num_threads(num_threads);

    double *augmented = (double *)malloc(n * 2 * n * sizeof(double));

    // Initialize the augmented matrix
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i * (2 * n) + j] = matrix[i * n + j]; // Original matrix
        }
        for (int j = n; j < 2 * n; j++) {
            augmented[i * (2 * n) + j] = (i == (j - n)) ? 1.0 : 0.0; // Identity matrix
        }
    }

    // Perform Gauss-Jordan elimination
    #pragma omp parallel
    {
        #pragma omp single nowait
        for (int i = 0; i < n; i++) {
            int pivot = i;
            #pragma omp task firstprivate(pivot)
            {
                // Find the pivot row
                for (int j = i + 1; j < n; j++) {
                    if (fabs(augmented[j * (2 * n) + i]) > fabs(augmented[pivot * (2 * n) + i])) {
                        pivot = j;
                    }
                }
                #pragma omp critical
                {
                    // Swap rows if needed
                    for (int k = 0; k < 2 * n; k++) {
                        double temp = augmented[i * (2 * n) + k];
                        augmented[i * (2 * n) + k] = augmented[pivot * (2 * n) + k];
                        augmented[pivot * (2 * n) + k] = temp;
                    }
                }
            }
            #pragma omp taskwait

            // Normalize the pivot row
            double pivotValue = augmented[i * (2 * n) + i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i * (2 * n) + j] /= pivotValue;
            }

            // Eliminate other rows
            #pragma omp task shared(augmented)
            {
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        double factor = augmented[j * (2 * n) + i];
                        for (int k = 0; k < 2 * n; k++) {
                            augmented[j * (2 * n) + k] -= factor * augmented[i * (2 * n) + k];
                        }
                    }
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    double *inverse = (double *)malloc(n * n * sizeof(double));
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i * n + j] = augmented[i * (2 * n) + j + n];
        }
    }

    // Free the augmented matrix
    free(augmented);

    return inverse;
}


int backsubstitution_p(double *A, double *b, double *x, int n)
{
    int t;
    printf("\nEnter number of threads: ");
    scanf("%d", &t);
    double temp;
    int i, j, k;

    for(i = 0; i < n-1; i++)
    {
        #pragma omp parallel default(none) num_threads(t) shared(n,A,b,i) private(j,k,temp)
        #pragma omp for schedule(static)
        for(j = i+1; j < n; j++)
        {
            temp = (A[j*(n)+i]) / (A[i*(n)+i]);
            for(k = i; k < n; k++)
            {
                A[j*(n)+k] -= temp * (A[i*(n)+k]);
            }
            b[j] -= temp * (b[i]);
        }
    }

    double tmp;
    #pragma omp parallel num_threads(t) default(none) private(i,j) shared(A, b, x, n, tmp)
    for(int i = n-1; i >= 0; i--)
    {
        #pragma omp single
        tmp = b[i];
        #pragma omp for reduction(+: tmp)
        for(j = i+1; j < n; j++)
            tmp += -A[i*n+j]*x[j];
        #pragma omp single
        x[i] = tmp / A[i*n+i];
    }

    //for(int i = 0; i < n; i++)
    //{
    //    printf("\t\t%lf\n", x[i]);
    //}

}


int conjugategradient_p(double *A, double *b, double *x, int n)
{
    int t;
    const int max_iterations = 50; // Fixed iterations

    printf("\nEnter number of threads: ");
    scanf("%d", &t);

    double r[n], p[n], px[n];
    #pragma omp parallel for num_threads(t)
    for(int i = 0; i < n; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }

    int q = max_iterations;
    double alpha = 0;
    while(q--)
    {
        double sum = 0;
        #pragma omp parallel for num_threads(t) reduction(+: sum)
        for(int i = 0; i < n; i++)
            sum += r[i] * r[i];

        double temp[n];
        #pragma omp parallel for num_threads(t)
        for(int i = 0; i < n; i++)
            temp[i] = 0;

        double num = 0;
        #pragma omp parallel for num_threads(t)
        for(int i = 0; i < n; i++)
        {
            #pragma omp parallel for reduction(+: temp[i])
            for(int j = 0; j < n; j++)
                temp[i] += A[i*n+j] * p[j];
        }

        #pragma omp parallel for num_threads(t) reduction(+: num)
        for(int j = 0; j < n; j++)
            num += temp[j] * p[j];

        alpha = sum / num;

        #pragma omp parallel for num_threads(t)
        for(int i = 0; i < n; i++)
        {
            px[i] = x[i];
            x[i] += alpha * p[i];
            r[i] -= alpha * temp[i];
        }

        double beta = 0;
        #pragma omp parallel for num_threads(t) reduction(+: beta)
        for(int i = 0; i < n; i++)
            beta += r[i] * r[i];

        beta /= sum;

        #pragma omp parallel for num_threads(t)
        for(int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];

        int c = 0;
        for(int i = 0; i < n; i++)
        {
            if(fabs(r[i]) < 0.000001)
                c++;
        }
        if(c == n)
            break;
    }

    //for(int i = 0; i < n; i++)
    //    printf("\t\t%f\n", x[i]);


}


int seidel_p(double A[], double b[], double x[], int n)
{
    const int maxit = 50; // Fixed iterations
    int i, j, k;
    double dxi;
    double epsilon = 1.0e-4;

    printf("\nEnter number of threads: ");
    int p;
    scanf("%d", &p);

    for(int i = 0; i < n; i++)
    {
        x[i] = 0;
    }

    for(k = 0; k < maxit; k++)
    {
        //printf("\n%d th iteration => \n", k+1);
        #pragma omp parallel for num_threads(p) schedule(static, n)
        for(i = 0; i < n; i++)
        {
            dxi = b[i];
            for(j = 0; j < n; j++)
            {
                if(j != i)
                    dxi -= A[i*n + j] * x[j];
                x[i] = dxi / A[i*n + i];
            }
            //printf("x %d = %f \n", i+1, x[i]);
        }
    }

}

int seidel_s(double A[], double b[], double x[], int n)
{
    int i,j,k;
    double dxi;
    double epsilon = 1.0e-4;
    int maxit = 50;
    double m[n];

    for(int i = 0; i<n; i++)
    {
        x[i] = 0;
    }
  //  printf("\nEnter number of iterations: ");
    //{
    //    scanf("%d", &maxit);
    //}

    for(k=0; k<maxit; k++)
    {
        double sum = 0.0;
       // printf("\n%d th iteration => \n", k+1);
        for(int i=0; i<n; i++)
        {
            dxi = b[i];
            for(int j=0; j<n; j++)
            {
                if(j!=i)
                {
                    dxi-=A[i*n + j] * x[j];
                }

                x[i] = dxi / A[i*n + i];
            }
          //  printf("x %d = %f \n", i+1, x[i]);
        }
    }
}



int conjugategradient_s(double *A, double *b, double *x,  int n)
{
    int max_iterations = 50;
   // printf("\nEnter number of iterations: ");
   // scanf("%d", &max_iterations);
    double r[n];
    double p[n];
    double px[n];
    for( int i = 0 ; i<n ; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }


    double alpha = 0;
    while(max_iterations--)
    {

        double sum = 0;
        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }

        double temp[n];
        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        double num = 0;
        for(int i = 0 ; i < n ; i++)
        {
            for(int j = 0 ; j < n ; j++ )
            {
                temp[i] = A[i*n+j]*p[j] + temp[i];
            }
        }
        for(int j = 0 ; j < n ; j++)
        {
            num = num + temp[j]*p[j];
        }

        alpha = sum / num;
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }
        double beta = 0;
        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }
        beta = beta / sum;
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }
        int c=0;
        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001 )
                c++;
        }
        if(c==n)
            break;
    }
    //for( int i = 0 ; i<n ; i++ )
    //    printf("\t\t%f\n", x[i]);

}


int backsubstitution_s(double *A, double *b, double *x, int n)
{
    int i, j, k;
    for(i =0; i < n-1; i++)
    {
        for(j = i+1; j < n; j++)
        {
                double temp = (A[j*(n)+i]) / (A[i*(n)+i]);

                for(k = i; k < n; k++)
                {
                    A[j*(n)+k] -= temp * (A[i*(n)+k]);
                }
                b[j] -= temp * (b[i]);
        }
    }
    double tmp;
    for(int i= n-1; i >=0; i--)
    {
        tmp = b[i];
        for(j = i+1; j< n; j++)
            tmp += -A[i*n+j]*x[j];
        x[i] = tmp/A[i*n+i];
    }

   // for(int i =0; i < n; i++)
   // {
   //     printf("\t\t%f\n",x[i]);
   // }

}

void l_u_d_s(float** a, float** l, float** u, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) {
                l[j][i] = 0;
            } else {
                l[j][i] = a[j][i];
                for (int k = 0; k < i; k++) {
                    l[j][i] -= l[j][k] * u[k][i];
                }
            }
        }
        for (int j = 0; j < n; j++) {
            if (j < i) {
                u[i][j] = 0;
            } else if (j == i) {
                u[i][j] = 1;
            } else {
                u[i][j] = a[i][j] / l[i][i];
                for (int k = 0; k < i; k++) {
                    u[i][j] -= (l[i][k] * u[k][j]) / l[i][i];
                }
            }
        }
    }
}

void l_u_d_p(float** a, float** l, float** u, int n)
{
    int t;
    // Ask for the number of threads
    printf("Enter number of threads: ");
    scanf("%d", &t);

    // Set the number of threads for OpenMP
    #pragma omp parallel num_threads(t) shared(a, l, u)
    {
        for (int i = 0; i < n; i++)
        {
            // LU Decomposition process for L and U
            // Parallel for the rows
            #pragma omp for
            for (int j = 0; j < n; j++)
            {
                if (j < i)
                {
                    l[j][i] = 0;
                    continue;
                }
                l[j][i] = a[j][i];
                for (int k = 0; k < i; k++)
                {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }

            // Parallel for the rows for U matrix
            #pragma omp for
            for (int j = 0; j < n; j++)
            {
                if (j < i)
                {
                    u[i][j] = 0;
                    continue;
                }
                if (j == i)
                {
                    u[i][j] = 1;
                    continue;
                }
                u[i][j] = a[i][j] / l[i][i];
                for (int k = 0; k < i; k++)
                {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}

void initialize_matrices(float*** a, float*** l, float*** u, int size) {
    // Allocate memory for the matrices
    *a = (float**) malloc(size * sizeof(float*));
    *l = (float**) malloc(size * sizeof(float*));
    *u = (float**) malloc(size * sizeof(float*));
   
    // Allocate memory for each row of the matrices
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        (*a)[i] = (float*) malloc(size * sizeof(float));
        (*l)[i] = (float*) malloc(size * sizeof(float));
        (*u)[i] = (float*) malloc(size * sizeof(float));
    }
}

// Fill the matrix with random values (done for matrix 'a')
void random_fill(float** matrix, int size) {
    // Fill matrix 'a' with random values
    if (VERBOSE) {
        printf("Producing random values\n");
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (float) ((rand() % 10) + 1);
        }
    }

    // Ensure the matrix is diagonal dominant to guarantee invertibility
    int diagCount = 0;
    float sum = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Sum all column values
            sum += fabs(matrix[i][j]);
        }
        // Remove the diagonal value from the sum
        sum -= fabs(matrix[i][diagCount]);
        // Add a random value to the sum and place in diagonal position
        matrix[i][diagCount] = sum + ((rand() % 5) + 1);
        ++diagCount;
        sum = 0;
    }
}
// Matrix multiplication validation function

 void matrix_validation(float** a, float** l, float** u, int size) {
    float** check = (float**) malloc(size * sizeof(float*));
    float** a2 = (float**) malloc(size * sizeof(float*));
    float** l2 = (float**) malloc(size * sizeof(float*));
    float** u2 = (float**) malloc(size * sizeof(float*));

    // Allocate memory for each row of the matrices
    for (int i = 0; i < size; ++i) {
        check[i] = (float*) malloc(size * sizeof(float));
    }
   
    a2 = a;
    l2 = l;
    u2 = u;

    // Matrix multiplication (u * l)
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            check[i][j] = 0;
            for (int k = 0; k < size; k++) {
                check[i][j] += u2[k][j] * l2[i][k];
            }
            check[i][j] = roundf(check[i][j]);
        }
    }

    if (VERBOSE) {
        printf("Check Matrix:\n");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                printf("%f ", check[i][j]);
            }
            printf("\n");
        }
    }

    // Validation of matrix multiplication
    int error = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (check[i][j] != a[i][j]) {
                error = 1;
                printf("Error at index (%d, %d): %f != %f\n", i, j, check[i][j], a[i][j]);
            }
        }
    }

    if (error == 1) {
        printf("Validation of matrix multiplication: Failed\n");
    } else {
        printf("Validation of matrix multiplication: Success\n");
    }

    // Free memory allocated for the matrices
    for (int i = 0; i < size; i++) {
        free(check[i]);
    }
    free(check);
    free(a2);
    free(l2);
    free(u2);
}



 
 int main() {
    int n;
    printf("Enter matrix size: ");
    scanf("%d", &n);

    // Allocate memory for LU Decomposition
    float **a, **l, **u;
    initialize_matrices(&a, &l, &u, n);
    random_fill(a, n);

    printf("\n1. LU Decomposition:\n");
    double td = omp_get_wtime();
    l_u_d_s(a, l, u, n);
    td = omp_get_wtime() - td;

    float **a2, **l2, **u2;
    initialize_matrices(&a2, &l2, &u2, n);
    random_fill(a2, n);

    double td1 = omp_get_wtime();
    l_u_d_p(a2, l2, u2, n);
    td1 = omp_get_wtime() - td1;

    printf("\nTime for serial execution (LU Decomposition): %0.30f\n", td);
    printf("Time for parallel execution (LU Decomposition): %0.30f\n", td1);

    // Validate LU Decomposition results (if applicable) for serial
    #ifdef VALIDATION
    matrix_validation(a, l, u, n);
    #endif

    // Allocate memory for the single-pointer matrix and check if allocation is successful
    double *matrix = (double *)malloc(n * n * sizeof(double));
    double *b = malloc(sizeof(double) * n);
    double *x = malloc(sizeof(double) * n);

    if (!matrix || !b || !x) {
        printf("Memory allocation failed!\n");
        exit(1);  // Exit the program if memory allocation fails
    }

    // Initialize matrices with random values
    for (int i = 0; i < (n * n); i++) {
        matrix[i] = (rand() / (double)RAND_MAX);
    }

    for (int i = 0; i < n; i++) {
        b[i] = (rand() / (double)RAND_MAX);
    }

    // Allocate additional matrices for different computations
    double *matrix2 = malloc(sizeof(double) * n * n);
    double *b2 = malloc(sizeof(double) * n);
    double *x2 = malloc(sizeof(double) * n);

    double *matrix3 = malloc(sizeof(double) * n * n);
    double *b3 = malloc(sizeof(double) * n);
    double *x3 = malloc(sizeof(double) * n);

    double *matrix4 = malloc(sizeof(double) * n * n);
    double *b4 = malloc(sizeof(double) * n);
    double *x4 = malloc(sizeof(double) * n);

    double *matrix5 = malloc(sizeof(double) * n * n);
    double *b5 = malloc(sizeof(double) * n);
    double *x5 = malloc(sizeof(double) * n);

    double *matrix6 = malloc(sizeof(double) * n * n);
    double *b6 = malloc(sizeof(double) * n);
    double *x6 = malloc(sizeof(double) * n);

    double *matrix7 = malloc(sizeof(double) * n * n);
    double *matrix8 = malloc(sizeof(double) * n * n);

    // Check if any memory allocation failed
    if (!matrix2 || !b2 || !x2 || !matrix3 || !b3 || !x3 || !matrix4 || !b4 || !x4 || !matrix5 || !b5 || !x5 || !matrix6 || !b6 || !x6 || !matrix7 || !matrix8) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // Copy data to new matrices
    memcpy(matrix2, matrix, sizeof(double) * n * n);
    memcpy(b2, b, sizeof(double) * n);

    memcpy(matrix3, matrix, sizeof(double) * n * n);
    memcpy(b3, b, sizeof(double) * n);

    memcpy(matrix4, matrix, sizeof(double) * n * n);
    memcpy(b4, b, sizeof(double) * n);

    memcpy(matrix5, matrix, sizeof(double) * n * n);
    memcpy(b5, b, sizeof(double) * n);

    memcpy(matrix6, matrix, sizeof(double) * n * n);
    memcpy(b6, b, sizeof(double) * n);

    memcpy(matrix7, matrix, sizeof(double) * n * n);
    memcpy(matrix8, matrix, sizeof(double) * n * n);

    // Time tracking variables
    double total_serial_time = 0.0;
    double total_parallel_time = 0.0;

    // Gauss Serial
    double start_time = omp_get_wtime();
    double *inverse = gaussJordanInverseSerial(matrix7, n);
    double end_time = omp_get_wtime();
    total_serial_time += (end_time - start_time);
    printf("\n2. Gauss-Jordan Inversion:\n");
    printf("\nTime for serial execution (Gauss-Jordan Inversion): %f seconds\n", end_time - start_time);

    // Gauss Parallel
    omp_set_num_threads(1);
    start_time = omp_get_wtime();
    inverse = gaussJordanInverse(matrix8, n);
    end_time = omp_get_wtime();
    total_parallel_time += (end_time - start_time);
    printf("\nTime for parallel execution (Gauss-Jordan Inversion): %f seconds\n", end_time - start_time);

    // Back Substitution
    printf("\n3. Back Substitution:\n");
    double ta = omp_get_wtime();
    backsubstitution_s(matrix, b, x, n);
    ta = omp_get_wtime() - ta;
    total_serial_time += ta;

    double ta1 = omp_get_wtime();
    backsubstitution_p(matrix2, b2, x2, n);
    ta1 = omp_get_wtime() - ta1;
    total_parallel_time += ta1;

    printf("\nTime for serial execution ( Back Substitution): %0.30f\n", ta);
    printf("Time for parallel execution (Back Substitution): %0.30f\n", ta1);

    // Conjugate Gradient
    printf("\n4. Conjugate Gradient: \n");
    double tb = omp_get_wtime();
    conjugategradient_s(matrix3, b3, x3, n);
    tb = omp_get_wtime() - tb;
    total_serial_time += tb;

    double tb1 = omp_get_wtime();
    conjugategradient_p(matrix4, b4, x4, n);
    tb1 = omp_get_wtime() - tb1;
    total_parallel_time += tb1;

    printf("\nTime for serial execution (Conjugate Gradient): %0.30f\n", tb);
    printf("Time for parallel execution (Conjugate Gradient): %0.30f\n", tb1);

    // Gauss Seidel
    printf("\n5. Gauss Seidel: \n");
    double tc = omp_get_wtime();
    seidel_s(matrix6, b6, x6, n);
    tc = omp_get_wtime() - tc;
    total_serial_time += tc;

    double tc1 = omp_get_wtime();
    seidel_p(matrix5, b5, x5, n);
    tc1 = omp_get_wtime() - tc1;
    total_parallel_time += tc1;

    printf("\nTime for serial execution (Gauss Seidel): %0.30f\n", tc);
    printf("Time for parallel execution (Gauss Seidel): %0.30f\n", tc1);

    // Print total execution times
    printf("\nTotal Serial Execution Time: %f seconds\n", total_serial_time);
    printf("Total Parallel Execution Time: %f seconds\n", total_parallel_time);

    // Free allocated memory
    free(inverse);
    free(matrix);
    free(b);
    free(x);
    free(matrix2);
    free(b2);
    free(x2);
    free(matrix3);
    free(b3);
    free(x3);
    free(matrix4);
    free(b4);
    free(x4);
    free(matrix5);
    free(b5);
    free(x5);
    free(matrix6);
    free(b6);
    free(x6);
    free(matrix7);
    free(matrix8);

    // Free LU Decomposition memory
    for (int i = 0; i < n; i++) {
        free(a[i]);
        free(l[i]);
        free(u[i]);
    }
    free(a);
    free(l);
    free(u);
    
        // Free LU Decomposition memory
    for (int i = 0; i < n; i++) {
        free(a2[i]);
        free(l2[i]);
        free(u2[i]);
    }
    free(a2);
    free(l2);
    free(u2);

}
