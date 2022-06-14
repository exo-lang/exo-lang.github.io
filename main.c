// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "example.h"

float* new_mat(int size, float value) {
  float* mat = malloc(size * sizeof(*mat));
  for (int i = 0; i < size; i++) {
    mat[i] = value;
  }
  return mat;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s M N K\n", argv[0]);
    return EXIT_FAILURE;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  if (M < 1 || N < 1 || K < 1) {
    printf("M, N, and K must all be positive!\n");
    return EXIT_FAILURE;
  }

  float* A = new_mat(M * K, 2.0);
  float* B = new_mat(K * N, 3.0);
  float* C = new_mat(M * N, 0.0);

  const int n_trials = 1000;

  clock_t start = clock();
  for (int i = 0; i < n_trials; i++) {
    example_sgemm(NULL, M, N, K, C, A, B);
  }
  clock_t end = clock();

  int msec = (end - start) * n_trials / CLOCKS_PER_SEC;

  printf("Each iteration ran in %d milliseconds\n", msec);
}