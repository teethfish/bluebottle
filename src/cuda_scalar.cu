
#include "cuda_scalar.h"

extern "C"
void cuda_part_scalar_malloc(void)
{
  if(scalar_on == 1){
    //TODO: store the coefficents of lebsque nodes in intial time
    // allocate device momery on host
    _parts_s = (part_struct_scalar**) malloc(nsubdom * sizeof(part_struct_scalar*));
    cpumem += nsubdom * sizeof(part_struct_scalar*);
    _anm_re = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _anm_im = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _anm_re0 = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);
    _anm_im0 = (real**) malloc(nsubdom * sizeof(real*));
    cpumem += nsubdom * sizeof(real*);

    // allocate device memory on device
    #pragma omp parallel num_threads(nsubdom)
    {
      int dev = omp_get_thread_num();
      (cudaSetDevice(dev + dev_start));

      (cudaMalloc((void**) &(_parts_s[dev]),
        sizeof(part_struct_scalar) * nparts));
      gpumem += sizeof(part_struct_scalar) * nparts;

      (cudaMalloc((void**) &(_anm_re[dev]),
        sizeof(real) * coeff_stride_scalar * nparts));
      gpumem += sizeof(real) * coeff_stride_scalar * nparts;
      (cudaMalloc((void**) &(_anm_im[dev]),
        sizeof(real) * coeff_stride_scalar * nparts));
      gpumem += sizeof(real) * coeff_stride_scalar * nparts;
      (cudaMalloc((void**) &(_anm_re0[dev]),
        sizeof(real) * coeff_stride_scalar * nparts));
      gpumem += sizeof(real) * coeff_stride_scalar * nparts;
      (cudaMalloc((void**) &(_anm_im0[dev]),
        sizeof(real) * coeff_stride_scalar * nparts));
      gpumem += sizeof(real) * coeff_stride_scalar * nparts;
    }
    // allocate the coefficent tabel on device
    (cudaMalloc((void**) &_nn_scalar, 25 * sizeof(int)));    
    gpumem += 25 * sizeof(int);
    (cudaMalloc((void**) &_mm_scalar, 25 * sizeof(int)));
    gpumem += 25 * sizeof(int);
  }
}

extern "C"
void cuda_part_scalar_push(void)
{
  if(scalar_on == 1) {
    // copy host data to device
    #pragma omp parallel num_threads(nsubdom)
    {
      int dev = omp_get_thread_num();
      (cudaSetDevice(dev + dev_start));

      (cudaMemcpy(_parts_s[dev], parts_s, sizeof(part_struct_scalar) * nparts,
         cudaMemcpyHostToDevice));
      (cudaMemcpy(_anm_re[dev], anm_re, sizeof(real) * coeff_stride_scalar
          * nparts, cudaMemcpyHostToDevice));
      (cudaMemcpy(_anm_im[dev], anm_im, sizeof(real) * coeff_stride_scalar
          * nparts, cudaMemcpyHostToDevice));
      (cudaMemcpy(_anm_re0[dev], anm_re0, sizeof(real) * coeff_stride_scalar
          * nparts, cudaMemcpyHostToDevice));
      (cudaMemcpy(_anm_im0[dev], anm_im0, sizeof(real) * coeff_stride_scalar
          * nparts, cudaMemcpyHostToDevice));
    }
    // copy coefficents to device

    // initialize coefficients table
    int nn_scalar[25] = {0,
                  1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4, 4, 4, 4, 4};
    int mm_scalar[25] = {0,
                  -1, 0, 1,
                  -2, -1, 0, 1, 2,
                  -3, -2, -1, 0, 1, 2, 3,
                  -4, -3, -2, -1, 0, 1, 2, 3, 4};
    (cudaMemcpy(_nn_scalar, nn_scalar, 25 * sizeof(int),
      cudaMemcpyHostToDevice));
    (cudaMemcpy(_mm_scalar, mm_scalar, 25 * sizeof(int),
      cudaMemcpyHostToDevice));
  }
}

extern "C"
void cuda_part_scalar_free(void)
{
  // free device memory on device
  if(scalar_on == 1) {
    #pragma omp parallel num_threads(nsubdom)
    {
      int dev = omp_get_thread_num();
      (cudaSetDevice(dev + dev_start));

      (cudaFree(_parts_s[dev]));
      (cudaFree(_anm_re[dev]));
      (cudaFree(_anm_im[dev]));
      (cudaFree(_anm_re0[dev]));
      (cudaFree(_anm_im0[dev]));
    }
    free(_parts_s);
    free(_anm_re);
    free(_anm_im);
    free(_anm_re0);
    free(_anm_im0);
  }
}


extern "C"
void cuda_part_scalar_pull(void)
{
  if(scalar_on == 1) {
    // all devices have the same particle data for now, so just copy one of them
    (cudaMemcpy(parts_s, _parts_s[0], sizeof(part_struct_scalar) * nparts,
      cudaMemcpyDeviceToHost));
    (cudaMemcpy(anm_re, _anm_re[0], sizeof(real) * coeff_stride_scalar
      * nparts,cudaMemcpyDeviceToHost));
    (cudaMemcpy(anm_im, _anm_im[0], sizeof(real) * coeff_stride_scalar
      * nparts,cudaMemcpyDeviceToHost));
    (cudaMemcpy(anm_re0, _anm_re0[0], sizeof(real) * coeff_stride_scalar
      * nparts,cudaMemcpyDeviceToHost));
    (cudaMemcpy(anm_im0, _anm_im0[0], sizeof(real) * coeff_stride_scalar
      * nparts,cudaMemcpyDeviceToHost));
  }
}

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
      scalar_explicit<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _s[dev], _conv_s[dev], _diff_s[dev], _u[dev], _v[dev], _w[dev], s_k, _dom[dev], dt);
    }
  }
}    

extern "C"
void cuda_show_variable(void)
{
  if(scalar_on == 1) {
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
      printf("before actual run show s0 and s\n");
      show_variable<<<numBlocks_s, dimBlocks_s>>>(_s0[dev], _s[dev], _dom[dev]);
      //printf("before actual run show s\n");
      //show_variable<<<numBlocks_s, dimBlocks_s>>>(_s[dev], _s[dev], _dom[dev]);
      //printf("before actual run show pressure\n");
      //show_variable<<<numBlocks_s, dimBlocks_s>>>(_p[dev], _p[dev], _dom[dev]);
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

extern "C"
void cuda_quad_interp_scalar(int dev, real *node_t, real *node_p, int nnodes, real *ss)
{
  int threads = nnodes;
  int blocks = nparts;

  dim3 dimBlocks(threads);
  dim3 numBlocks(blocks);

  if(nparts > 0) {
    interpolate_nodes_scalar<<<numBlocks, dimBlocks>>>(_s[dev], _parts[dev], _parts_s[dev], _dom[dev], node_t, node_p, nnodes, ss, bc_s);
  }
}


extern "C"
void cuda_scalar_lamb(void)
{
  //interpolate the outer field to Lebsque nodes and use the value to calculate the coefficents
  // TODU:set up those table and constant number into global memory
  #pragma omp parallel num_threads(nsubdom)
  {
    int dev = omp_get_thread_num();
    (cudaSetDevice(dev + dev_start));

    // set up Lebsque nodes info
    int i;  // iterator

    /*
    // set up coefficient table
    int nn[25] = {0,
                  1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4, 4, 4, 4, 4};
    int mm[25] = {0,
                  -1, 0, 1,
                  -2, -1, 0, 1, 2,
                  -3, -2, -1, 0, 1, 2, 3,
                  -4, -3, -2, -1, 0, 1, 2, 3, 4};
    */
  
    // set up quadrature nodes for 7th-order Lebedev quadrature
    real PI14 = 0.25 * PI;
    real PI12 = 0.5 * PI;
    real PI34 = 0.75 * PI;
    real PI54 = 1.25 * PI;
    real PI32 = 1.5 * PI;
    real PI74 = 1.75 * PI;
    real alph1 = 0.955316618124509; //54.736
    real alph2 = 2.186276035465284; //125.264

    // weights
    real A1 = 0.598398600683775;//0.159572960182342;//0.119679720136760;//
    real A2 = 0.478718880547015;//0.283685262546377;//0.403919055461543;//
    real A3 = 0.403919055461543;//0.265071880146639;//0.403919055461543;
    real B = 0.;//0.253505610897313;//0.359039160410267;

    // nodes TODO: find a more elegant way of fixing the divide by sin(0)
    // TODO: put this in GPU constant memory
    real a1_t[6] = {PI12, PI12, PI12, PI12, 0.+DIV_ST, PI-DIV_ST};
    //real a1_t[6] = {PI12, PI12, PI12, PI12, 0., PI};
    real a1_p[6] = {0., PI12, PI, PI32, 0., 0.};
    real a2_t[12] = {PI12, PI12, PI12, PI12,
                     PI14, PI14, PI14, PI14,
                     PI34, PI34, PI34, PI34};
    real a2_p[12] = {PI14, PI34, PI54, PI74,
                     0., PI12, PI, PI32,
                     0., PI12, PI, PI32};
    real a3_t[8] = {alph1, alph1, alph1, alph1,
                    alph2, alph2, alph2, alph2};
    real a3_p[8] = {PI14, PI34, PI54, PI74,
                    PI14, PI34, PI54, PI74};

    int nnodes = 26;
    // put all quadrature nodes together for interpolation
    real node_t[nnodes];
    real node_p[nnodes];
    for(i = 0; i < 6; i++) {
      node_t[i] = a1_t[i];
      node_p[i] = a1_p[i];
    }
    for(i = 0; i < 12; i++) {
      node_t[6+i] = a2_t[i];
      node_p[6+i] = a2_p[i];
    }
    for(i = 0; i < 8; i++) {
      node_t[18+i] = a3_t[i];
      node_p[18+i] = a3_p[i];
    }

    /* ALREADY MADE IT INTO GLOBAL VARIABLE
    // create a place to temporarily store field variables at quadrature nodes
    int *_nn;
    int *_mm;
    (cudaMalloc((void**) &_nn, nnodes * sizeof(int)));
    gpumem += nnodes * sizeof(int);
    (cudaMalloc((void**) &_mm, nnodes * sizeof(int)));
    gpumem += nnodes * sizeof(int);
    (cudaMemcpy(_nn, nn, nnodes * sizeof(int),
      cudaMemcpyHostToDevice));
    (cudaMemcpy(_mm, mm, nnodes * sizeof(int),
      cudaMemcpyHostToDevice));
    */

    real *_node_t;
    real *_node_p;
    (cudaMalloc((void**) &_node_t, nnodes * sizeof(real)));
    gpumem += nnodes * sizeof(real);
    (cudaMalloc((void**) &_node_p, nnodes * sizeof(real)));
    gpumem += nnodes * sizeof(real);
    (cudaMemcpy(_node_t, node_t, nnodes * sizeof(real),
      cudaMemcpyHostToDevice));
    (cudaMemcpy(_node_p, node_p, nnodes * sizeof(real),
      cudaMemcpyHostToDevice));


    real *_ss;
    (cudaMalloc((void**) &_ss, nnodes * nparts * sizeof(real)));
    gpumem += nnodes * nparts * sizeof(real);
         
    // interpolate field scalar to Lebsque nodes
    // TODU: double check cuda_quad_check_nodes
    // cuda_quad_check_nodes(dev, _node_t, _node_p, nnodes);
    cuda_quad_interp_scalar(dev, _node_t, _node_p, nnodes, _ss);

    // create temporary storage for inner product inergrands
    real *int_scalar_re;
    real *int_scalar_im;
    (cudaMalloc((void**) &int_scalar_re, 
      nparts * nnodes * coeff_stride_scalar * sizeof(real)));
    gpumem += nparts * nnodes * coeff_stride_scalar * sizeof(real);
    (cudaMalloc((void**) &int_scalar_im, 
      nparts * nnodes * coeff_stride_scalar * sizeof(real)));
    gpumem += nparts * nnodes * coeff_stride_scalar * sizeof(real);

    // calculate the coefficients
    dim3 dimBlocks(nnodes);
    dim3 numBlocks(nparts, coeff_stride_scalar);
    if(nparts > 0) {
      cuda_get_coeffs_scalar<<<numBlocks, dimBlocks>>>(_parts[dev],_parts_s[dev], _nn_scalar, _mm_scalar, _node_t, _node_p, _ss, coeff_stride_scalar, _anm_re[dev], _anm_re0[dev], _anm_im[dev], _anm_im0[dev], int_scalar_re, int_scalar_im, nnodes, A1, A2, A3, B);
    }

    // clean up temporary variables
    (cudaFree(_ss));
    (cudaFree(_node_t));
    (cudaFree(_node_p));
    (cudaFree(int_scalar_re));
    (cudaFree(int_scalar_im));
  }
}

extern "C"
real cuda_scalar_lamb_err(void)
{
  real *errors = (real*) malloc(nsubdom * sizeof(real));
  // cpumem += nsubdom * sizeof(real);
  real max = FLT_MIN;
 
  if(scalar_on == 1) {
    if(nparts > 0) {
      #pragma omp parallel num_threads(nsubdom)
      {
        int dev = omp_get_thread_num();
        (cudaSetDevice(dev + dev_start));

        // create a place to store sorted coefficients and errors
        real *part_errors = (real*) malloc(nparts * sizeof(real));
        real *_sorted_coeffs;
        real *_sorted_errors;
        real *_part_errors; 
        (cudaMalloc((void**) &_sorted_coeffs,
          nparts*2*coeff_stride*sizeof(real)));
        gpumem += 2 * nparts * coeff_stride * sizeof(real);
        (cudaMalloc((void**) &_sorted_errors,
          nparts*2*coeff_stride*sizeof(real)));
        gpumem += 2 * nparts * coeff_stride * sizeof(real);
        (cudaMalloc((void**) &_part_errors,
          nparts*sizeof(real)));
        gpumem += nparts * sizeof(real);

        // sort the coefficients and calculate errors along the way
        dim3 dimBlocks(1);
        dim3 numBlocks(nparts);

        compute_error_scalar<<<numBlocks, dimBlocks>>>(lamb_cut_scalar, coeff_stride_scalar, nparts, _anm_re[dev], _anm_re0[dev], _anm_im[dev], _anm_im0[dev], _sorted_coeffs, _sorted_errors, _part_errors, _dom[dev]);

        // copy the errors back to device
        (cudaMemcpy(part_errors, _part_errors,
          nparts*sizeof(real), cudaMemcpyDeviceToHost));
        
        // find maximum error of all particles
        real tmp = FLT_MIN;
        for(int i = 0; i < nparts; i++) {
          if(part_errors[i] > tmp) tmp = part_errors[i];
        }
        errors[dev] = tmp;
        // clean up
        (cudaFree(_sorted_coeffs));
        (cudaFree(_sorted_errors));
        (cudaFree(_part_errors));

        // store copy of coefficients for future calculation
        (cudaMemcpy(_anm_re0[dev], _anm_re[dev],
          coeff_stride_scalar*nparts*sizeof(real), cudaMemcpyDeviceToDevice));
        (cudaMemcpy(_anm_im0[dev], _anm_im[dev],
          coeff_stride_scalar*nparts*sizeof(real), cudaMemcpyDeviceToDevice));

        // clean up
        free(part_errors);
      }
      
      // find maximum error of all subdomains
      for(int i = 0; i < nsubdom; i++) {
        if(errors[i] > max) max = errors[i];
      }
      // clean up
      free(errors);

      return max;
    } else return 0.;
  } else return 0.;
}

extern "C"
void cuda_part_BC_scalar(void)
{
/*
  int n = 0;
  int m = 0;
  int CC;
  int P;
  for(int i = Dom.Gcc.is; i < Dom.Gcc.ie; i++){
    for(int j = Dom.Gcc.js; j < Dom.Gcc.je; j++){
      for(int k = Dom.Gcc.ks; k < Dom.Gcc.ke; k++){
        CC = i + j*Dom.Gcc.s1b + k*Dom.Gcc.s2b;
        P = phase[CC];
        if(P > -1){
          m++;
          if(phase_shell[CC]==0){
            n++;
          }
        }
      }
    }
  }
  printf("m, n is %d %d\n",m, n);
*/
  // using the lamb's coefficents calculated previously to apply the inner boundary condition for scalar equation
  if(scalar_on == 1) {
    if(nparts > 0) {
      #pragma omp parallel num_threads(nsubdom)
      {
        int dev = omp_get_thread_num();
        cudaSetDevice(dev + dev_start);

        int threads_c = MAX_THREADS_DIM;
        int blocks_y = 0;
        int blocks_z = 0;

        blocks_y = (int)ceil((real) dom[dev].Gcc.jn / (real) threads_c);
        blocks_z = (int)ceil((real) dom[dev].Gcc.kn / (real) threads_c);

        dim3 dimBlocks_c(threads_c, threads_c);
        dim3 numBlocks_c(blocks_y, blocks_z);
        part_BC_scalar<<<numBlocks_c, dimBlocks_c>>>(_s0[dev], _phase[dev], _phase_shell[dev], _parts[dev], _parts_s[dev], _dom[dev], coeff_stride_scalar, _anm_re[dev], _anm_im[dev]);
      }
    }
  }
}

