#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  // auto start = CycleTimer::currentSeconds();
  thread_local double *minDist = new double[args->end - args->start];
  
  // Initialize arrays
  for (int m = args->start; m < args->end; m++) {
    minDist[m - args->start] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = 0; k < args->K; k++) {
    for (int m = args->start; m < args->end; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m - args->start]) {
        minDist[m - args->start] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  // auto end = CycleTimer::currentSeconds();
  // printf("%s: cost: %lf ms\n", __func__, (end - start) * 1000.0);

  // free(minDist);
}

void computeAssignmentsParallel(WorkerArgs* const args, int n_workers) {
  auto start = CycleTimer::currentSeconds();
  WorkerArgs* all_args = static_cast<WorkerArgs*>(malloc(n_workers * sizeof(WorkerArgs)));

  int span = args->M / n_workers;
  for (int i = 0; i < n_workers; i++) {
    all_args[i] = *args;
    all_args[i].start = i * span;
    all_args[i].end = std::min((i + 1) * span, args->M);
  }

  std::vector<std::thread> threads(n_workers - 1);
  for (int i = 1; i < n_workers; i++) {
    threads[i - 1] = std::thread(computeAssignments, &all_args[i]);
  }

  computeAssignments(&all_args[0]);

  for (auto& t : threads) {
    t.join();
  }

  free(all_args);
  auto end = CycleTimer::currentSeconds();
  printf("%s: cost: %lf ms\n", __func__, (end - start) * 1000.0);
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  auto start = CycleTimer::currentSeconds();
  thread_local int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  // free(counts);

  auto end = CycleTimer::currentSeconds();
  printf("%s: cost: %lf ms\n", __func__, (end - start) * 1000.0);
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  auto start = CycleTimer::currentSeconds();
  thread_local double *accum = new double[args->K];

  // Zero things out
  memset(accum, 0, args->K * sizeof(double));

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  // free(accum);

  auto end = CycleTimer::currentSeconds();
  // printf("%s: cost: %lf ms\n", __func__, (end - start) * 1000.0);
}

void computeCostParallel(WorkerArgs * const args, int n_workers) {
  auto start = CycleTimer::currentSeconds();
  thread_local double *accum = new double[args->K * n_workers];

  // Zero things out
  memset(accum, 0, args->K * n_workers * sizeof(double));

  // Sum cost for all data points assigned to centroid
  auto worker_func = [] (WorkerArgs* worker_arg, double* accumulate) {
    int N = worker_arg->N;
    for (int m = worker_arg->start; m < worker_arg->end; m++) {
      int k = worker_arg->clusterAssignments[m];
      accumulate[k] += dist(&worker_arg->data[m * N],
                      &worker_arg->clusterCentroids[k * N], N);
    }
  };

  int span = args->M / n_workers;
  std::vector<WorkerArgs> all_args(n_workers);
  for (int i = 0; i < n_workers; i++) {
    all_args[i] = *args ;
    all_args[i].start = span * i;
    all_args[i].end = std::min(span * (i + 1), args->M);
  }

  std::vector<std::thread> workers(n_workers - 1);
  for (int i = 1; i < n_workers; i++) {
    workers[i - 1] = std::thread(worker_func, &all_args[i], accum + i * args->K);
  }

  worker_func(&all_args[0], accum);

  for (auto &t : workers) {
    t.join();
  }

  // Update costs
  for (int t = 1; t < n_workers; t++) {
    for (int k = 0; k < args->K; k++) {
      accum[k] += accum[k + t];
    }
  }

  memcpy(args->currCost, accum, args->K * sizeof(double));

  // free(accum);

  auto end = CycleTimer::currentSeconds();
  printf("%s: cost: %lf ms\n", __func__, (end - start) * 1000.0);
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  int n_workers = 8;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    args.start = 0;
    args.end = K;

    // computeAssignments(&args);
    computeAssignmentsParallel(&args, n_workers);

    computeCentroids(&args);

    // computeCost(&args);
    computeCostParallel(&args, n_workers);

    iter++;
    printf("----> iter = %d <----\n", iter);
  }

  free(currCost);
  free(prevCost);
}
