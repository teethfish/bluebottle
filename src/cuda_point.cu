/*******************************************************************************
 ********************************* BLUEBOTTLE **********************************
 *******************************************************************************
 *
 *  Copyright 2012 - 2016 Adam Sierakowski, The Johns Hopkins University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Please contact the Johns Hopkins University to use Bluebottle for
 *  commercial and/or for-profit applications.
 ******************************************************************************/

#include "cuda_point.h"

#include <cuda.h>

extern "C"
void cuda_point_malloc(void)
{
  // allocate device memory on host
  _points = (point_struct**) malloc(nsubdom * sizeof(point_struct*));
  cpumem += nsubdom * sizeof(point_struct*);

  // allocate device memory on device
  #pragma omp parallel num_threads(nsubdom)
  {
    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));

    (cudaMalloc((void**) &(_points[dev]),
      sizeof(point_struct) * npoints));
    gpumem += sizeof(point_struct) * npoints;
  }
}

extern "C"
void cuda_point_push(void)
{
  // copy host data to device
  #pragma omp parallel num_threads(nsubdom)
  {
    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));

    (cudaMemcpy(_points[dev], points, sizeof(point_struct) * npoints,
      cudaMemcpyHostToDevice));
  }
}

extern "C"
void cuda_point_pull(void)
{
  // all devices have the same particle data for now, so just copy one of them
  (cudaMemcpy(points, _points[0], sizeof(point_struct) * npoints,
    cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_point_free(void)
{
  free(_points);
}

void cuda_move_points(void)
{
  // parallelize over CPU threads
  #pragma omp parallel num_threads(nsubdom)
	{
		int dev = omp_get_thread_num();
		(cudaSetDevice(dev + dev_start));

		int threads = MAX_THREADS_1D;
		int blocks = (int)ceil((real) npoints / (real) threads); 
    if(threads > npoints) {
      threads = npoints;
      blocks = 1;
    }
		
		dim3 dimBlocks(threads);
		dim3 numBlocks(blocks);

		move_points<<<numBlocks, dimBlocks>>>(_points[dev], npoints, _u[dev], _v[dev], _w[dev], _dom[dev], mu, dt);
	}
}


