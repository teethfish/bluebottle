

#include "cuda_scalar.h"
//#include "scalar_kernel.h"

// scalar; west
// PERIODIC boundary condition
__global__ void BC_s_W_P(real *s, dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    s[dom->Gcc._isb + tj*s1b + tk*s2b] = s[(dom->Gcc._ie-1) + tj*s1b + tk*s2b]; 
}

// Dirichlet bc: a known value of scalar on the surface
__global__ void BC_s_W_D(real *s, dom_struct *dom, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb)) {
    s[dom->Gcc._isb + tj*s1b + tk*s2b] = 2.* bc_s - s[(dom->Gcc._is) + tj*s1b + tk*s2b];
  }
}

// Neumann bc: constant heat flux bc, F_n = q_s(constant)
__global__ void BC_s_W_N(real *s, dom_struct *dom, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    s[dom->Gcc._isb + tj*s1b + tk*s2b] = s[(dom->Gcc._is) + tj*s1b + tk*s2b] - bc_s*dom->dx;
}

// E plane
// Periodic boundary condition
__global__ void BC_s_E_P(real *s, dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    s[(dom->Gcc._ie) + tj*s1b + tk*s2b] = s[dom->Gcc._is + tj*s1b + tk*s2b];
}

// DIRICHLET bc
__global__ void BC_s_E_D(real *s, dom_struct *dom, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb)) {
    s[(dom->Gcc._ie) + tj*s1b + tk*s2b] = 2. * bc_s - s[(dom->Gcc._ie-1) + tj*s1b + tk*s2b];
  }
}

__global__ void BC_s_E_N(real *s, dom_struct *dom, real bc_s)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb)) {
    s[dom->Gcc._ie + tj*s1b + tk*s2b] = s[(dom->Gcc._ie-1) + tj*s1b + tk*s2b] - dom->dx*bc_s;
  }
}

// N surface
// scalar, periodic boundary condition
__global__ void BC_s_N_P(real *s, dom_struct *dom)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    s[ti + (dom->Gcc._jeb-1)*s1b + tk*s2b] = s[ti + dom->Gcc._js*s1b + tk*s2b];
}

// Dirichelt bc
__global__ void BC_s_N_D(real *s, dom_struct *dom, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb)){
    s[ti + (dom->Gcc._je)*s1b + tk*s2b] = 2*bc_s - s[ti + (dom->Gcc._je-1)*s1b + tk*s2b];
  }
}

// Neumann bc
__global__ void BC_s_N_N(real *s, dom_struct *dom, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    s[ti + (dom->Gcc._je)*s1b + tk*s2b] = s[ti + (dom->Gcc._je-1)*s1b + tk*s2b] - dom->dy*bc_s;
}

// S plane
//periodic bc
__global__ void BC_s_S_P(real *s, dom_struct *dom)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    s[ti + dom->Gcc._jsb*s1b + tk*s2b] = s[ti + (dom->Gcc._je-1)*s1b + tk*s2b];
}

// Dirichlet bc
__global__ void BC_s_S_D(real *s, dom_struct *dom, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb)){
    s[ti + dom->Gcc._jsb*s1b + tk*s2b] = 2*bc_s - s[ti + dom->Gcc._js*s1b + tk*s2b];
  }
}

// Neumann bc
__global__ void BC_s_S_N(real *s, dom_struct *dom, real bc_s)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    s[ti + dom->Gcc._jsb*s1b + tk*s2b] = s[ti + dom->Gcc._js*s1b + tk*s2b] - bc_s*dom->dy;
}

// B plane
// Periodic bc
__global__ void BC_s_B_P(real *s, dom_struct *dom)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    s[ti + tj*s1b + dom->Gcc._ksb*s2b] = s[ti + tj*s1b + (dom->Gcc._ke-1)*s2b];
}

// Dirichelt
__global__ void BC_s_B_D(real *s, dom_struct *dom, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    s[ti + tj*s1b + dom->Gcc._ksb*s2b] = 2*bc_s - s[ti + tj*s1b + dom->Gcc._ks*s2b];
}

// Neumann bc
__global__ void BC_s_B_N(real *s, dom_struct *dom, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    s[ti + tj*s1b + dom->Gcc._ksb*s2b] = s[ti + tj*s1b + dom->Gcc._ks*s2b] - bc_s*dom->dz;
}

// T plane
// Periodic bc
__global__ void BC_s_T_P(real *s, dom_struct *dom)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    s[ti + tj*s1b + (dom->Gcc._keb-1)*s2b] = s[ti + tj*s1b + dom->Gcc._ks*s2b];
}

// Dirichlet bc
__global__ void BC_s_T_D(real *s, dom_struct *dom, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    s[ti + tj*s1b + (dom->Gcc._ke)*s2b] = 2*bc_s - s[ti + tj*s1b + (dom->Gcc._ke-1)*s2b];
}

// Neumann bc
__global__ void BC_s_T_N(real *s, dom_struct *dom, real bc_s)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    s[ti + tj*s1b + (dom->Gcc._ke)*s2b] = s[ti + tj*s1b + (dom->Gcc._ke-1)*s2b] - bc_s*dom->dz;
}


__global__ void scalar_explicit(real *s, real *conv_s, real *diff_s, real *s0, real *u, real *v, real *w, real s_k, dom_struct *dom, real dt)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  // working constants
  real ddx = 1. / dom->dx;
  real ddy = 1. / dom->dy;
  real ddz = 1. / dom->dz;

  // loop over x-plane
  if(tj < dom->Gcc._je && tk < dom->Gcc._ke) {
    for(int i = dom->Gcc._is; i < dom->Gcc._ie; i++) {
      int C   = i       + tj      *dom->Gcc._s1b + tk      *dom->Gcc._s2b;
      int Cx0 = (i - 1) + tj      *dom->Gcc._s1b + tk      *dom->Gcc._s2b;
      int Cx1 = (i + 1) + tj      *dom->Gcc._s1b + tk      *dom->Gcc._s2b;
      int Cy0 = i       + (tj - 1)*dom->Gcc._s1b + tk      *dom->Gcc._s2b;
      int Cy1 = i       + (tj + 1)*dom->Gcc._s1b + tk      *dom->Gcc._s2b;
      int Cz0 = i       + tj      *dom->Gcc._s1b + (tk - 1)*dom->Gcc._s2b;
      int Cz1 = i       + tj      *dom->Gcc._s1b + (tk + 1)*dom->Gcc._s2b;
      int fx0 = i       + tj      *dom->Gfx._s1b + tk      *dom->Gfx._s2b;
      int fx1 = (i + 1) + tj      *dom->Gfx._s1b + tk      *dom->Gfx._s2b;
      int fy0 = i       + tj      *dom->Gfy._s1b + tk      *dom->Gfy._s2b;
      int fy1 = i       + (tj + 1)*dom->Gfy._s1b + tk      *dom->Gfy._s2b;
      int fz0 = i       + tj      *dom->Gfz._s1b + tk      *dom->Gfz._s2b;
      int fz1 = i       + tj      *dom->Gfz._s1b + (tk + 1)*dom->Gfz._s2b;

      // calculate the convection term
      real convec_x = u[fx1] * 0.5 * (s0[Cx1] + s0[C]) - u[fx0] * 0.5 * (s0[C] + s0[Cx0]);
      convec_x = convec_x * ddx;
      real convec_y = v[fy1] * 0.5 * (s0[Cy1] + s0[C]) - v[fy0] * 0.5 * (s0[C] + s0[Cy0]);
      convec_y = convec_y * ddy;
      real convec_z = w[fz1] * 0.5 * (s0[Cz1] + s0[C]) - w[fz0] * 0.5 * (s0[C] + s0[Cz0]);
      convec_z = convec_z * ddz;  
      conv_s[C] = convec_x + convec_y + convec_z;
      
      // calculate the diffusion term
      real diff_x = s_k * (s0[Cx0] - 2.*s0[C] + s0[Cx1]) * ddx * ddx;
      real diff_y = s_k * (s0[Cy0] - 2.*s0[C] + s0[Cy1]) * ddy * ddy;
      real diff_z = s_k * (s0[Cz0] - 2.*s0[C] + s0[Cz1]) * ddz * ddz;
      diff_s[C] = diff_x + diff_y + diff_z;

      // calculate s at current time
      s[C] = s0[C] + dt * (diff_s[C] - conv_s[C]);
    
      /*if(s[C] > 50.0){
        printf("s[%d] is %f, i,j,k is %d %d %d, convection is %f, diffusion is %f\n", C, s[C], i, tj, tk, conv_s[C], diff_s[C]);
        printf("convx, convy convz, diffx, diffy, diffz is %f %f %f %f %f %f\n",convec_x, convec_y, convec_z, diff_x, diff_y, diff_z); 
      }*/
    }
  }
}

__global__ void update_scalar(real *s, real *s0, dom_struct *dom)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  // loop over x-plane
  if(tj < dom->Gcc._je && tk < dom->Gcc._ke) {
    for(int i = dom->Gcc._is; i < dom->Gcc._ie; i++) {
      // update current field as previous step
      int C = i + tj*dom->Gcc._s1b + tk*dom->Gcc._s2b;
      s0[C] = s[C];
    }
  }
}
    



































