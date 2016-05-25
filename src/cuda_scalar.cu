
#include "cuda_scalar.h"

extern "C"
void cuda_scalar_malloc(void)
{
  if(scalar_on == 1) {
    // allocate device memory on host
    _s0 = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _s = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _conv0_s = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _conv_s = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _diff0_s = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _diff_s = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    // allocate device memory on device
    #pragma omp parallel num_threads(nsubdom)
    {
      int dev = omp_get_thread_num();
      (cudaSetDevice(dev + dev_start));

      (cudaMalloc((void**) &(_s0[dev]),sizeof(real) * dom[dev].Gcc.s3b));
      gpumem += dom[dev].Gcc.s3b * sizeof(real);

      (cudaMalloc((void**) &(_s[dev]),sizeof(real) * dom[dev].Gcc.s3b));
      gpumem += dom[dev].Gcc.s3b * sizeof(real);

      (cudaMalloc((void**) &(_conv0_s[dev]),sizeof(real) * dom[dev].Gcc.s3b));
      gpumem += dom[dev].Gcc.s3b * sizeof(real);

      (cudaMalloc((void**) &(_conv_s[dev]),sizeof(real) * dom[dev].Gcc.s3b));
      gpumem += dom[dev].Gcc.s3b * sizeof(real);
    
      (cudaMalloc((void**) &(_diff0_s[dev]),sizeof(real) * dom[dev].Gcc.s3b));
      gpumem += dom[dev].Gcc.s3b * sizeof(real);
      
      (cudaMalloc((void**) &(_diff_s[dev]),sizeof(real) * dom[dev].Gcc.s3b));
      gpumem += dom[dev].Gcc.s3b * sizeof(real);
    }
  }
}

extern "C"
void cuda_scalar_push(void)
{
  if(scalar_on == 1) {
  // copy host data to device
  #pragma omp parallel num_threads(nsubdom)
  {
    int i, j, k;          // iterators
    int ii, jj, kk;       // helper iterators
    int C, CC;            // cell references
    
    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));

    // set up host working arrays for subdomain copy from host to device
    real *ss = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    real *ss0 = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    // select appropriate subdomain
    // s
    for(k = dom[dev].Gcc.ksb; k < dom[dev].Gcc.keb; k++) {
      for(j = dom[dev].Gcc.jsb; j < dom[dev].Gcc.jeb; j++) {
        for(i = dom[dev].Gcc.isb; i < dom[dev].Gcc.ieb; i++) {
          ii = i - dom[dev].Gcc.isb;
          jj = j - dom[dev].Gcc.jsb;
          kk = k - dom[dev].Gcc.ksb;
          C = i + j * Dom.Gcc.s1b + k * Dom.Gcc.s2b;
          CC = ii + jj * dom[dev].Gcc.s1b + kk * dom[dev].Gcc.s2b;
          ss[CC] = s[C];
          ss0[CC] = s0[C];
        }
      }
    }

    // copy from host to device
    (cudaMemcpy(_s0[dev], ss0, sizeof(real) * dom[dev].Gcc.s3b,
      cudaMemcpyHostToDevice));
    (cudaMemcpy(_s[dev], ss, sizeof(real) * dom[dev].Gcc.s3b,
      cudaMemcpyHostToDevice));

    //free host subdomain working arrays
    free(ss);
    free(ss0);
  }
  }
}

extern "C"
void cuda_scalar_pull(void)
{
  if(scalar_on == 1) {
  // copy device data to host
  #pragma omp parallel num_threads(nsubdom)
  {
    int i, j, k;          // iterators
    int ii, jj, kk;       // helper iterators
    int C, CC;            // cell references

    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));

    // host working arrays for subdomain copy from device to host
    real *ss = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    real *ss0 = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));

    // copy from device to host
    (cudaMemcpy(ss0, _s0[dev], sizeof(real) * dom[dev].Gcc.s3b,
      cudaMemcpyDeviceToHost));
    (cudaMemcpy(ss, _s[dev], sizeof(real) * dom[dev].Gcc.s3b,
      cudaMemcpyDeviceToHost));
  
    // scalar
    for(k = dom[dev].Gcc.ksb; k < dom[dev].Gcc.keb; k++) {
      for(j = dom[dev].Gcc.jsb; j < dom[dev].Gcc.jeb; j++) {
        for(i = dom[dev].Gcc.isb; i < dom[dev].Gcc.ieb; i++) {
          ii = i - dom[dev].Gcc.isb;
          jj = j - dom[dev].Gcc.jsb;
          kk = k - dom[dev].Gcc.ksb;
          C = i + j * Dom.Gcc.s1b + k * Dom.Gcc.s2b;
          CC = ii + jj * dom[dev].Gcc.s1b + kk * dom[dev].Gcc.s2b;
          s0[C] = ss0[CC];
          s[C] = ss[CC];
          //divU[C] = pdivU[CC];
        }
      }
    }    
    
    // free host subdomain working arrays
    free(ss);
    free(ss0);
  }
  }
}

extern "C"
void cuda_scalar_free(void)
{
  if(scalar_on == 1){
  // free device memory on device
  #pragma omp parallel num_threads(nsubdom)
  {
    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));
    
    (cudaFree(_s0[dev]));
    (cudaFree(_s[dev]));
    (cudaFree(_conv0_s[dev]));
    (cudaFree(_conv_s[dev]));
    (cudaFree(_diff0_s[dev]));
    (cudaFree(_diff_s[dev]));
  }
  // free device memory on host
  free(_s0);
  free(_s);
  free(_conv0_s);
  free(_conv_s);
  free(_diff0_s);
  free(_diff_s);

  }
}


extern "C"
void cuda_scalar_BC(void)
{
  if(scalar_on == 1){
  // CPU threading for multi-GPU
  #pragma omp parallel num_threads(nsubdom)
  {
    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));

    int threads_x = 0;
    int threads_y = 0;
    int threads_z = 0;
    int blocks_x = 0;
    int blocks_y = 0;
    int blocks_z = 0;

    // check whether each subdomain boundary (E, W, N, S, T, B) is
    // an external boundary
    if(dom[dev].W == -1) {
      // set up kernel call
      // scalar field
      if(dom[dev].Gcc.jnb < MAX_THREADS_DIM)
        threads_y = dom[dev].Gcc.jnb;
      else
        threads_y = MAX_THREADS_DIM;

      if(dom[dev].Gcc.knb < MAX_THREADS_DIM)
        threads_z = dom[dev].Gcc.knb;
      else
        threads_z = MAX_THREADS_DIM;

      blocks_y = (int)ceil((real) dom[dev].Gcc.jnb / (real) threads_y);
      blocks_z = (int)ceil((real) dom[dev].Gcc.knb / (real) threads_z);

      dim3 dimBlocks_s(threads_y, threads_z);
      dim3 numBlocks_s(blocks_y, blocks_z);

      // apply BC to scalar field on W face
      switch(bc_s.sW) {
        case PERIODIC:
          BC_s_W_P<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev]);
          break;
        case DIRICHLET:
          BC_s_W_D<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sWD);
          break;
        case NEUMANN:
          BC_s_W_N<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sWN);
          break;
      }
    }
    if(dom[dev].E == -1) {
      // scalar
      if(dom[dev].Gcc.jnb < MAX_THREADS_DIM)
        threads_y = dom[dev].Gcc.jnb;
      else
        threads_y = MAX_THREADS_DIM;

      if(dom[dev].Gcc.knb < MAX_THREADS_DIM)
        threads_z = dom[dev].Gcc.knb;
      else
        threads_z = MAX_THREADS_DIM;

      blocks_y = (int)ceil((real) dom[dev].Gcc.jnb / (real) threads_y);
      blocks_z = (int)ceil((real) dom[dev].Gcc.knb / (real) threads_z);

      dim3 dimBlocks_s(threads_y, threads_z);
      dim3 numBlocks_s(blocks_y, blocks_z);
 
      // apply BC to scalar field on E face
      switch(bc_s.sE) {
        case PERIODIC:
          BC_s_E_P<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev]);
          break;
        case DIRICHLET:
          BC_s_E_D<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sED);
          break;
        case NEUMANN:
          BC_s_E_N<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sEN);
          break;
      }
    }  
    if(dom[dev].S == -1) {
      // scalar
      if(dom[dev].Gcc.knb < MAX_THREADS_DIM)
        threads_z = dom[dev].Gcc.knb;
      else
        threads_z = MAX_THREADS_DIM;

      if(dom[dev].Gcc.inb < MAX_THREADS_DIM)
        threads_x = dom[dev].Gcc.inb;
      else
        threads_x = MAX_THREADS_DIM;

      blocks_z = (int)ceil((real) dom[dev].Gcc.knb / (real) threads_z);
      blocks_x = (int)ceil((real) dom[dev].Gcc.inb / (real) threads_x);

      dim3 dimBlocks_s(threads_z, threads_x);
      dim3 numBlocks_s(blocks_z, blocks_x);
  
      // apply BC to scalar field on S face
      switch(bc_s.sS) {
        case PERIODIC:
          BC_s_S_P<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev]);
          break;
        case DIRICHLET:
          BC_s_S_D<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sSD);
          break;
        case NEUMANN:
          BC_s_S_N<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sSN);
          break;
      }      
    }
    if(dom[dev].N == -1) {
      // scalar
      if(dom[dev].Gcc.knb < MAX_THREADS_DIM)
        threads_z = dom[dev].Gcc.knb;
      else
        threads_z = MAX_THREADS_DIM;

      if(dom[dev].Gcc.inb < MAX_THREADS_DIM)
        threads_x = dom[dev].Gcc.inb;
      else
        threads_x = MAX_THREADS_DIM;

      blocks_z = (int)ceil((real) dom[dev].Gcc.knb / (real) threads_z);
      blocks_x = (int)ceil((real) dom[dev].Gcc.inb / (real) threads_x);

      dim3 dimBlocks_s(threads_z, threads_x);
      dim3 numBlocks_s(blocks_z, blocks_x);
    
      // apply BC to scalar field on S face
      switch(bc_s.sN) {
        case PERIODIC:
          BC_s_N_P<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev]);
          break;
        case DIRICHLET:
          BC_s_N_D<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sND);
          break;
        case NEUMANN:
          BC_s_N_N<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sNN);
          break;
      }
    }
    if(dom[dev].B == -1) {
      // scalar
      if(dom[dev].Gcc.inb < MAX_THREADS_DIM)
        threads_x = dom[dev].Gcc.inb;
      else
        threads_x = MAX_THREADS_DIM;

      if(dom[dev].Gcc.jnb < MAX_THREADS_DIM)
        threads_y = dom[dev].Gcc.jnb;
      else
        threads_y = MAX_THREADS_DIM;

      blocks_x = (int)ceil((real) dom[dev].Gcc.inb / (real) threads_x);
      blocks_y = (int)ceil((real) dom[dev].Gcc.jnb / (real) threads_y);

      dim3 dimBlocks_s(threads_x, threads_y);
      dim3 numBlocks_s(blocks_x, blocks_y);

      // apply BC to scalar field on B face
      switch(bc_s.sB) {
        case PERIODIC:
          BC_s_B_P<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev]);
          break;
        case DIRICHLET:
          BC_s_B_D<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sBD);
          break;
        case NEUMANN:
          BC_s_B_N<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sBN);
          break;
      }
    }
    if(dom[dev].T == -1) {
      // scalar 
      if(dom[dev].Gcc.inb < MAX_THREADS_DIM)
        threads_x = dom[dev].Gcc.inb;
      else
        threads_x = MAX_THREADS_DIM;

      if(dom[dev].Gcc.jnb < MAX_THREADS_DIM)
        threads_y = dom[dev].Gcc.jnb;
      else
        threads_y = MAX_THREADS_DIM;

      blocks_x = (int)ceil((real) dom[dev].Gcc.inb / (real) threads_x);
      blocks_y = (int)ceil((real) dom[dev].Gcc.jnb / (real) threads_y);

      dim3 dimBlocks_s(threads_x, threads_y);
      dim3 numBlocks_s(blocks_x, blocks_y);

      // apply BC to scalar field on T face
      switch(bc_s.sT) {
        case PERIODIC:
          BC_s_T_P<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev]);
          break;
        case DIRICHLET:
          BC_s_T_D<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sTD);
          break;
        case NEUMANN:
          BC_s_T_N<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _dom[dev], bc_s.sTN);
          break;
      }
    }
  }
  }
}

extern "C"
void cuda_solve_scalar_explicit(void)
{
  if(scalar_on == 1){
    #pragma omp parallel num_threads(nsubdom)
    {
      int dev = omp_get_thread_num();
      cudaSetDevice(dev + dev_start);

      int threads_y = 0;
      int threads_z = 0;
      int blocks_y = 0;
      int blocks_z = 0;

      if(dom[dev].Gcc._jn < MAX_THREADS_DIM)
        threads_y = dom[dev].Gcc._jn;
      else
        threads_y = MAX_THREADS_DIM;

      if(dom[dev].Gcc._kn < MAX_THREADS_DIM)
        threads_z = dom[dev].Gcc._kn;
      else
        threads_z = MAX_THREADS_DIM;


      blocks_y = (int)ceil((real) dom[dev].Gcc._jn / (real) threads_y);
      blocks_z = (int)ceil((real) dom[dev].Gcc._kn / (real) threads_z);

      dim3 dimBlocks_s(threads_y, threads_z);
      dim3 numBlocks_s(blocks_y, blocks_z);


      scalar_explicit<<<numBlocks_s, dimBlocks_s>>>(_s[dev], _conv_s[dev], _diff_s[dev], _s0[dev], _u[dev], _v[dev], _w[dev],s_k, _dom[dev], dt);
    }
  }
}    

extern "C"
void cuda_update_scalar(void)
{
  if(scalar_on == 1){
    #pragma omp parallel num_threads(nsubdom)
    {
      int dev = omp_get_thread_num();
      cudaSetDevice(dev + dev_start);

      int threads_y = 0;
      int threads_z = 0;
      int blocks_y = 0;
      int blocks_z = 0;

      if(dom[dev].Gcc._jn < MAX_THREADS_DIM)
        threads_y = dom[dev].Gcc._jn;
      else
        threads_y = MAX_THREADS_DIM;

      if(dom[dev].Gcc._kn < MAX_THREADS_DIM)
        threads_z = dom[dev].Gcc._kn;
      else
        threads_z = MAX_THREADS_DIM;


      blocks_y = (int)ceil((real) dom[dev].Gcc._jn / (real) threads_y);
      blocks_z = (int)ceil((real) dom[dev].Gcc._kn / (real) threads_z);

      dim3 dimBlocks_s(threads_y, threads_z);
      dim3 numBlocks_s(blocks_y, blocks_z);

      update_scalar<<<numBlocks_s, dimBlocks_s>>>(_s[dev], _s0[dev], _dom[dev]);
    }
  }
}


    



          




















