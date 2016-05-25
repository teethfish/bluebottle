

#ifndef _CUDA_SCALAR_H
#define _CUDA_SCALAR_H


extern "C"
{
#include "bluebottle.h"
#include "scalar.h"
}


__global__ void BC_s_W_P(real *s, dom_struct *dom);

__global__ void BC_s_W_D(real *s, dom_struct *dom, real bc_s);

__global__ void BC_s_W_N(real *s, dom_struct *dom, real bc_s);

// E plane
__global__ void BC_s_E_P(real *s, dom_struct *dom);

__global__ void BC_s_E_D(real *s, dom_struct *dom, real bc_s);

__global__ void BC_s_E_N(real *s, dom_struct *dom, real bc_s);

// N plane
__global__ void BC_s_N_P(real *s, dom_struct *dom);

__global__ void BC_s_N_D(real *s, dom_struct *dom, real bc_s);

__global__ void BC_s_N_N(real *s, dom_struct *dom, real bc_s);

// S plane
__global__ void BC_s_S_P(real *s, dom_struct *dom);

__global__ void BC_s_S_D(real *s, dom_struct *dom, real bc_s);

__global__ void BC_s_S_N(real *s, dom_struct *dom, real bc_s);

// B plane
__global__ void BC_s_B_P(real *s, dom_struct *dom);

__global__ void BC_s_B_D(real *s, dom_struct *dom, real bc_s);

__global__ void BC_s_B_N(real *s, dom_struct *dom, real bc_s);

// T plane
__global__ void BC_s_T_P(real *s, dom_struct *dom);

__global__ void BC_s_T_D(real *s, dom_struct *dom, real bc_s);

__global__ void BC_s_T_N(real *s, dom_struct *dom, real bc_s);

// calculate the scalar field explicitly
__global__ void scalar_explicit(real *s, real *conv_s, real *diff_s, real *s0, real *u, real *v, real *w, real s_k, dom_struct *dom, real dt);


// update scalar filed
__global__ void update_scalar(real *s, real *s0, dom_struct *dom);


#endif



