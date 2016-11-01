/*******************************************************************************
 ********************************* BLUEBOTTLE **********************************
 *******************************************************************************
 *
 *  Copyright 2015 - 2016 Yayun Wang, The Johns Hopkins University
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

__global__ void scalar_explicit(real *s0, real *s, real *conv_s, real *diff_s, real *conv0_s, real *diff0_s, real *u, real *v, real *w, real s_D, dom_struct *dom, real dt, real dt0)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  // working constants
  real ab0 = 0.5 * dt / dt0;  // for Adams-Bashforth stepping
  real ab = 1. + ab0;         // for Adams-Bashforth stepping
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
      // current time step
      conv_s[C] = convec_x + convec_y + convec_z;

      // calculate the diffusion term
      real diff_x = s_D * (s0[Cx0] - 2.*s0[C] + s0[Cx1]) * ddx * ddx;
      real diff_y = s_D * (s0[Cy0] - 2.*s0[C] + s0[Cy1]) * ddy * ddy;
      real diff_z = s_D * (s0[Cz0] - 2.*s0[C] + s0[Cz1]) * ddz * ddz;
      diff_s[C] = diff_x + diff_y + diff_z;

      // Adams-Bashforth
      if(dt0 > 0) {
        s[C] = s0[C] + dt * (ab * diff_s[C] - ab0 * diff0_s[C] - (ab * conv_s[C] - ab0 * conv0_s[C]));
      }
      else {
        s[C] = s0[C] + dt * (diff_s[C] - conv_s[C]);
      }

      /*if(s[C] > 150.0 || s[C] < -10.){
        printf("s[%d] is %f, i,j,k is %d %d %d, conv,diff is %f %f; is %f %f %f %f %f %f\n", C, s[C], i, tj, tk, conv_s[C], diff_s[C],s0[Cx0],s0[Cx1],s0[Cy0],s0[Cy1], s0[Cz0], s0[Cz1]);
      }*/

    }
  }
}

__global__ void show_variable(real *s0, real *s, dom_struct *dom)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  if(tj < dom->Gcc._je && tk < dom->Gcc._ke) {
    for(int i = dom->Gcc._is; i < dom->Gcc._ie; i++) {
      int C   = i       + tj      *dom->Gcc._s1b + tk      *dom->Gcc._s2b;
      if(C > 65600 && C < 65700){
        printf("diff0[%d,%d,%d] and diff[%d] is %f %f\n", i, tj, tk, C, s0[C], s[C]);
      }
    }
  }
}


__global__ void update_scalar(real *s, real *s0, real *conv_s, real *conv0_s, real *diff_s, real *diff0_s, dom_struct *dom)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x + DOM_BUF;
  int tk = blockIdx.y * blockDim.y + threadIdx.y + DOM_BUF;

  // loop over x-plane
  if(tj < dom->Gcc._je && tk < dom->Gcc._ke) {
    for(int i = dom->Gcc._is; i < dom->Gcc._ie; i++) {
      // update current field as previous step
      int C = i + tj*dom->Gcc._s1b + tk*dom->Gcc._s2b;
      s0[C] = s[C];
      conv0_s[C] = conv_s[C];
      diff0_s[C] = diff_s[C];
    }
  }
}

__global__ void check_nodes_scalar(int nparts, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, real *theta, real *phi, int nnodes, BC_s bc_s)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  // convert node (r, theta, phi) to (x, y, z)
  real xp, yp, zp;  // Cartesian radial vector
  real x, y, z;   // Cartesian location of node
  rtp2xyz(parts_s[part].rs*parts[part].r, theta[node], phi[node], &xp, &yp, &zp);

  // shift from particle center
  x = xp + parts[part].x;
  y = yp + parts[part].y;
  z = zp + parts[part].z;

  // start off with all -1's
  parts[part].nodes[node] = -1;

  // check if the node is interfered with by a wall
  // set equal to some number to identify which wall is interefering
  // TODO:here re-use the parts[part].nodes[ndoe], which first been used in function check_nodes;
  if(x - dom->xs < 0) {
    if(bc_s.sW == DIRICHLET)
      if(parts[part].nodes[node] == -1)
        parts[part].nodes[node] = -10;
    } if(x - dom->xe > 0) {
      if(bc_s.sE == DIRICHLET)
        if(parts[part].nodes[node] == -1)
          parts[part].nodes[node] = -11;
    } if(y - dom->ys < 0) {
      if(bc_s.sS == DIRICHLET)
        if(parts[part].nodes[node] == -1)
          parts[part].nodes[node] = -12;
    } if(y - dom->ye > 0) {
      if(bc_s.sN == DIRICHLET)
        if(parts[part].nodes[node] == -1)
          parts[part].nodes[node] = -13;
    } if(z - dom->zs < 0) {
      if(bc_s.sB == DIRICHLET)
        if(parts[part].nodes[node] == -1)
          parts[part].nodes[node] = -14;
    } if(z - dom->ze > 0) {
      if(bc_s.sT == DIRICHLET)
        if(parts[part].nodes[node] == -1)
          parts[part].nodes[node] = -15;
    }
}
  

__global__ void interpolate_nodes_scalar(real *s, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, real *theta, real *phi, int nnodes, real *ss,  BC_s bc_s)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  real ddx = 1. / dom->dx;
  real ddy = 1. / dom->dy;
  real ddz = 1. / dom->dz;

  // convert node (r, theta, phi) to (x, y, z)
  real xp, yp, zp;  // Cartesian radial vector
  real x, y, z;   // Cartesian location of node
  rtp2xyz(parts_s[part].rs*parts[part].r, theta[node], phi[node], &xp, &yp, &zp);

  // shift from particle center
  x = xp + parts[part].x;
  y = yp + parts[part].y;
  z = zp + parts[part].z;

  if(x < dom->xs && bc_s.sW == PERIODIC) x = x + dom->xl;
  else if(x > dom->xe && bc_s.sE == PERIODIC) x = x - dom->xl;
  if(y < dom->ys && bc_s.sS == PERIODIC) y = y + dom->yl;
  else if(y > dom->ye && bc_s.sN == PERIODIC) y = y - dom->yl;
  if(z < dom->zs && bc_s.sB == PERIODIC) z = z + dom->zl;
  else if(z > dom->ze && bc_s.sT == PERIODIC) z = z - dom->zl;

  __syncthreads();

  // find index of cell containing node
  int i = floor((x - dom->xs) * ddx) + DOM_BUF;
  int j = floor((y - dom->ys) * ddy) + DOM_BUF;
  int k = floor((z - dom->zs) * ddz) + DOM_BUF;
  if(i < dom->Gcc.is) i = dom->Gcc.is;
  if(j < dom->Gcc.js) j = dom->Gcc.js;
  if(k < dom->Gcc.ks) k = dom->Gcc.ks;
  if(i > dom->Gcc.ie-1) i = dom->Gcc.ie-1;
  if(j > dom->Gcc.je-1) j = dom->Gcc.je-1;
  if(k > dom->Gcc.ke-1) k = dom->Gcc.ke-1;
  int C = i + j*dom->Gcc.s1b + k*dom->Gcc.s2b;
  // Cartesian location of center of cell
  real xx = (i-0.5) * dom->dx + dom->xs;
  real yy = (j-0.5) * dom->dy + dom->ys;
  real zz = (k-0.5) * dom->dz + dom->zs;


  // interpolate scalar
  real s_c = s[C];
  real s_w = s[C-1];
  real s_e = s[C+1];
  real s_s = s[C-dom->Gcc.s1b];
  real s_n = s[C+dom->Gcc.s1b];
  real s_b = s[C-dom->Gcc.s2b];
  real s_t = s[C+dom->Gcc.s2b];
  real dsdx = 0.5*(s_e - s_w) * ddx;
  real dsdy = 0.5*(s_n - s_s) * ddy;
  real dsdz = 0.5*(s_t - s_b) * ddz;
  // assumption that sclar is a constant on the particle surface
  ss[node+nnodes*part] = s_c + dsdx*(x-xx) + dsdy*(y-yy) + dsdz*(z-zz) - parts_s[part].s;

 // printf("ss[%d] is %f, s_c,dsdx, dsdy, dsdz is %f %f %f %f\n", node+nnodes*part, ss[node+nnodes*part], s_c, dsdx, dsdy, dsdz);
  // wall temperature if this node intersects wall
  real sswall = (parts[part].nodes[node] == -10)*bc_s.sWD
            + (parts[part].nodes[node] == -11)*bc_s.sED
            + (parts[part].nodes[node] == -12)*bc_s.sSD
            + (parts[part].nodes[node] == -13)*bc_s.sND
            + (parts[part].nodes[node] == -14)*bc_s.sBD
            + (parts[part].nodes[node] == -15)*bc_s.sTD;
  ss[node+nnodes*part] = (parts[part].nodes[node]==-1)*ss[node+part*nnodes]
                        + (parts[part].nodes[node] < -1)*sswall;
  //printf("ss[%d] is %f\n", node+nnodes*part, ss[node+nnodes*part]);
}
    
__global__ void cuda_get_coeffs_scalar(part_struct *parts, part_struct_scalar *parts_s, int *nn, int *mm, real *node_t, real *node_p, real *ss, int stride_scalar, real *anm_re, real *anm_re0, real *anm_im, real *anm_im0, real *int_scalar_re, real *int_scalar_im, int nnodes, real A1, real A2, real A3, real B)
{
  // TODO:not correct right now, needs to check the integration and coeff_stride, ncoeffs, order and so on
  int node = threadIdx.x;
  int part = blockIdx.x;
  int coeff = blockIdx.y;
  real asr = 1.0 / parts_s[part].rs;
  real rsa = parts_s[part].rs;
  int i;

  if(coeff < parts_s[part].ncoeff) {
    // calculate integrand at each node
    int j = part*nnodes*stride_scalar + coeff*nnodes + node;
    int_scalar_re[j] = 0.0;
    int_scalar_im[j] = 0.0; 
    int n = nn[coeff];
    int m = mm[coeff];
    real theta = node_t[node];
    real phi = node_p[node];

    real N_nm = nnm(n,m);
    real P_nm = pnm(n,m,theta);

    int_scalar_re[j] = N_nm*P_nm*ss[node+part*nnodes]*cos(m*phi);
    int_scalar_im[j] = -N_nm*P_nm*ss[node+part*nnodes]*sin(m*phi);
    //printf("ss[%d] is %f\n",node+part*nnodes,ss[node+part*nnodes]);

    __syncthreads();
    if(node == 0) {
      int_scalar_re[j] *= A1;
      int_scalar_im[j] *= A1;
      for(i = 1; i < 6; i++) {
        int_scalar_re[j] += A1 * int_scalar_re[j+i];
        int_scalar_im[j] += A1 * int_scalar_im[j+i];
      }
      for(i = 6; i < 18; i++) {
        int_scalar_re[j] += A2 * int_scalar_re[j+i];
        int_scalar_im[j] += A2 * int_scalar_im[j+i];
      }
      for(i = 18; i < 26; i++) {
        int_scalar_re[j] += A3 * int_scalar_re[j+i];
        int_scalar_im[j] += A3 * int_scalar_im[j+i];
      }
      real A = pow(rsa,n) - pow(asr,n+1.);
      anm_re[stride_scalar*part+coeff] = int_scalar_re[j]/ A;
      anm_im[stride_scalar*part+coeff] = int_scalar_im[j]/ A;

/*       
      if(stride_scalar*part+coeff == 0){
        printf("anm_re[%d] is %f\n", stride_scalar*part+coeff, anm_re[stride_scalar*part+coeff]);
       // printf("ss[%d] is %f\n", node+part*nnodes, ss[node+part*nnodes]);
      }
*/      
    }
  }
} 
        
__global__ void cuda_get_coeffs_scalar_perturbation(part_struct_scalar *parts_s, int *nn, int *mm, real *node_t, real *node_p, int stride_scalar, real *anm_re_perturb, real *anm_im_perturb, real *int_scalar_re, real *int_scalar_im, int nnodes, real A1, real A2, real A3, real dt, real s_D)
{
  int node = threadIdx.x;
  int part = blockIdx.x;
  int coeff = blockIdx.y;
  int i;

  if(coeff < parts_s[part].ncoeff) {
    // calculate integrand at each node
    int j = part*nnodes*stride_scalar + coeff*nnodes + node;
    int_scalar_re[j] = 0.0;
    int_scalar_im[j] = 0.0;
    int n = nn[coeff];
    int m = mm[coeff];
    real theta = node_t[node];
    real phi = node_p[node];

    real N_nm = nnm(n,m);
    real P_nm = pnm(n,m,theta);

    int_scalar_re[j] = N_nm * P_nm * (parts_s[part].s - parts_s[part].s0) / dt * cos(m*phi);
    int_scalar_im[j] = -N_nm * P_nm * (parts_s[part].s - parts_s[part].s0) / dt * sin(m*phi);
/* 
    int_scalar_re[j] = N_nm * P_nm * (-100.*1.*sin(1.*dt)) * cos(m*phi);
    int_scalar_im[j] = -N_nm * P_nm * (-100.*1.*sin(1.*dt)) * sin(m*phi);
    //printf("dt is %f and dTdt is %f\n", dt, -10000.*sin(100.*dt));
*/ 
    __syncthreads();
    if(node == 0) {
      int_scalar_re[j] *= A1;
      int_scalar_im[j] *= A1;
      for(i = 1; i < 6; i++) {
        int_scalar_re[j] += A1 * int_scalar_re[j+i];
        int_scalar_im[j] += A1 * int_scalar_im[j+i];
      }
      for(i = 6; i < 18; i++) {
        int_scalar_re[j] += A2 * int_scalar_re[j+i];
        int_scalar_im[j] += A2 * int_scalar_im[j+i];
      }
      for(i = 18; i < 26; i++) {
        int_scalar_re[j] += A3 * int_scalar_re[j+i];
        int_scalar_im[j] += A3 * int_scalar_im[j+i];
      }
      anm_re_perturb[stride_scalar*part+coeff] = int_scalar_re[j]/s_D;
      anm_im_perturb[stride_scalar*part+coeff] = int_scalar_im[j]/s_D;
 
/*     
      if(stride_scalar*part+coeff == 0){
        printf("anm_re_perturb[%d] is %f\n", stride_scalar*part+coeff, anm_re_perturb[stride_scalar*part+coeff]);
        //printf("dT0dt is %f\n", (parts_s[part].s - parts_s[part].s0)/dt);
      }
*/    
    }
  }
}



__global__ void compute_error_scalar(real lamb_cut, int stride, int nparts, real *anm_re, real *anm_re0, real *anm_im, real *anm_im0, real *coeffs, real *errors, real *part_errors, dom_struct *dom)
{
  int part = blockIdx.x;
  int i,j;
  real tmp = FLT_MIN;
  int loc = 0;
  //real avg = 0;
  real div = 0;

  // create shared memory space
  __shared__ real s_coeffs[2*25];  // ** have to hard-code this length **
  __shared__ real s_coeffs0[2*25]; // ** have to hard-code this length **
                            // using 2 coefficient sets, each holding
                            // a maximum of 25 coefficients (4th-order
                            // truncation)

  // copy coeffs for this particle into shared memory
  for(i = 0; i < stride; i++) {
    s_coeffs[i] = anm_re[part*stride+i];
    s_coeffs[i+1*stride] = anm_im[part*stride+i];
    s_coeffs0[i] = anm_re0[part*stride+i];
    s_coeffs0[i+1*stride] = anm_im0[part*stride+i];
  }

  // sort the coefficients in shared memory and calculate errors along the way
  for(i = 0; i < 2*stride; i++) {
    // search for the largest magnitude value in shared and store its location
    tmp = FLT_MIN;
    for(j = 0; j < 2*stride; j++) {
      if(s_coeffs[j]*s_coeffs[j] > tmp) {
        tmp = s_coeffs[j]*s_coeffs[j];
        loc = j;
      }
    }

    // move the largest value into sorted list
    coeffs[part*stride*2+i] = s_coeffs[loc];

    // if its corresponding coefficient has large enough magnitude,
    // compute error for this coefficient
    if(fabs(s_coeffs[loc]) > lamb_cut*fabs(coeffs[part*stride*2+0])) {
      div = fabs(s_coeffs[loc]);// + fabs(avg)*1e-4;
      if(div < 1e-16) div = 1e-16;
      errors[part*stride*2+i] = fabs((s_coeffs[loc] - s_coeffs0[loc]) / div);
    } else errors[part*stride*2+i] = 0.;

    // discard this value since we've used it once
    s_coeffs[loc] = 0.;

  }
  
  // find the largest error for each particle
  tmp = FLT_MIN;
  for(i = 0; i < 2*stride; i++) {
    if(errors[part*2*stride+i] > tmp) {
      tmp = errors[part*stride*2+i];
    }
  }

  // write error to return for each particle
  part_errors[part] = tmp;
  //printf("final erros for part %d is %f\n", part, part_errors[part]);
}

__global__ void part_BC_scalar(real *s, int *phase, int *phase_shell, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, int stride, real *anm_re, real *anm_im, real *anm_re00, real *anm_im00, real *anm_re_perturb, real *anm_im_perturb, real s_D, real perturbation, real dt)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x + dom->Gcc._js;
  int tk = blockDim.y*blockIdx.y + threadIdx.y + dom->Gcc._ks;
  //int C;
  int CC;
  real x, y, z;         // pressure node location Cartesian
  real X, Y, Z;         // particle position
  real r, theta, phi;   // velocity node location spherical
  real ss_tmp;        // temporary pressure
  int P;            // particle number
  real a;               // particle radius
  int order;            // particle order for scalar lamb coefficents
  real s_surface;       // particle surface tmeperature

  if(tj < dom->Gcc._je && tk < dom->Gcc._ke) {
    for(int i = dom->Gcc._is; i < dom->Gcc._ie; i++) {
      CC = i + tj*dom->Gcc._s1b + tk*dom->Gcc._s2b;
      //C = (i-DOM_BUF) + (tj-DOM_BUF)*dom->Gcc._s1 + (tk-DOM_BUF)*dom->Gcc._s2;
      // get particle number
      P = phase[CC];
      if(P > -1) {
        a = parts[P].r;
        X = parts[P].x;
        Y = parts[P].y;
        Z = parts[P].z;
        order = parts_s[P].order;
        s_surface = parts_s[P].s;
      
        x = (i-0.5) * dom->dx + dom->xs - X;
        if(x <= 2.*a-dom->xl) x += dom->xl;
        if(x >= dom->xl-2.*a) x -= dom->xl;
        y = (tj-0.5) * dom->dy + dom->ys - Y;
        if(y <= 2.*a-dom->yl) y += dom->yl;
        if(y >= dom->yl-2.*a) y -= dom->yl;
        z = (tk-0.5) * dom->dz + dom->zs - Z;
        if(z <= 2.*a-dom->zl) z += dom->zl;
        if(z >= dom->zl-2.*a) z -= dom->zl;
        
        xyz2rtp(x, y, z, &r, &theta, &phi);
        // calculate analytic solution
        real ar = a / r;
        real ra = r / a;
        ss_tmp = (pow(ra,0.0) - pow(ar,1.)) * X_an(0, theta, phi, anm_re, anm_im, P, stride);
        for(int n = 1; n <= order; n++) {
          ss_tmp += (pow(ra,n) - pow(ar,n+1)) * X_an(n, theta, phi, anm_re, anm_im, P, stride);
        }

        // if perturbation not zero, add the perturbation correction term
        if(perturbation > 0) {
          real rs = parts_s[P].rs * parts[P].r; //integration surface
          real tmp = 0.0;
          real tmp1 = 0.0; //dT_0/dt term
          real tmp2 = 0.0;   // d(A_lm)dt term; r dependence associate with each order n
          real denominator = 0.0; // coefficients associate with each order n

          for(int n = 0; n <= order; n++) {
            denominator = 1./((2*n+1)*(pow(a, 2*n+1) - pow(rs, 2*n+1)));
            tmp1 = (inte_M(n,rs,a) + pow(a, 2*n+1)*inte_N(n,a,r) - pow(rs,2*n+1)*inte_N(n,rs,r)) * denominator * pow(r, n);
            tmp1 -= (pow(rs*a,2*n+1)*inte_N(n,a,rs) + pow(a,2*n+1)*inte_M(n,rs,r) - pow(rs,2*n+1)*inte_M(n,a,r)) * denominator *  pow(1.0/r, n+1);
            tmp1 = tmp1 * X_an(n, theta, phi, anm_re_perturb, anm_im_perturb, P, stride);

            tmp2 = (inte_B(n,a,rs,a) + pow(a,2*n+1) * inte_A(n,a,a,r) - pow(rs,2*n+1)*inte_A(n,a,rs,r)) * denominator * pow(r, n);
            tmp2 -= (pow(rs*a,2*n+1) * inte_A(n,a,a,rs) + pow(a, 2*n+1)*inte_B(n,a,rs,r) - pow(rs, 2*n+1)*inte_B(n,a,a,r)) * denominator * pow(1./r, n+1);
            // second part of solution, d(A_lm)dt
            tmp2 *= perturbation_X_an(n, theta, phi, anm_re, anm_re00, anm_im, anm_im00, P, stride, dt)/s_D;             
            tmp = tmp1 + tmp2;            
            ss_tmp += tmp;
            //if(phase_shell[CC] < 1) printf("ss_tmp, tmp1, tmp2, tmp, n is %f %f %f %f %d\n", ss_tmp, tmp1, tmp2, tmp, n);
          }
        }
        ss_tmp += s_surface;
        // only apply value at nodes inside particle & phase_shell==0
        // phase_shell = 1 means normal nodes, phase_shell = 0 means pressure nodes
        s[CC] = ss_tmp * (phase[CC] > -1 && phase_shell[CC] < 1) + (phase_shell[CC] > 0)*s_surface; 
        //if(phase_shell[CC] < 1)printf("s[CC] is %f\n", s[CC]);
      }
    }
  }
}

__device__ real X_an(int n, real theta, real phi, real *anm_re, real *anm_im, int pp, int stride)
{
  int coeff = 0;
  for(int j = 0; j < n; j++) coeff += 2*j + 1;

  coeff = coeff + pp*stride;

  real sum = 0.0;
  for(int m = -n; m <= n; m++) {
    sum += nnm(n,m)*pnm(n,m,theta)*(anm_re[coeff]*cos(m*phi) - anm_im[coeff]*sin(m*phi));
    coeff++;
  }
  return sum;
}

__device__ real perturbation_X_an(int n, real theta, real phi, real *anm_re, real *anm_re0, real *anm_im, real *anm_im0, int pp, int stride, real dt)
{
  int coeff = 0;
  for(int j = 0; j < n; j++) coeff += 2*j + 1;

  coeff = coeff + pp*stride;

  real sum = 0.0;

  for(int m = -n; m <= n; m++) {
    sum += nnm(n,m)*pnm(n,m,theta)*((anm_re[coeff]-anm_re0[coeff])*cos(m*phi) - (anm_im[coeff] - anm_im0[coeff])*sin(m*phi));
    //printf("anm_re[coeff]-anm_re0[coeff] is %f\n",anm_re[coeff]-anm_re0[coeff]);
    coeff++;
  }
  return sum/dt;
}

__device__ real inte_A(int n, real a, real r0, real r1)
{
  real sum = 0.0;
  sum = (r1*r1 - r0*r0)*0.5 / pow(a,n) - 1.0/(1-2*n) * pow(a, n+1)*(pow(r1,1-2*n) - pow(r0, 1-2*n));
  return sum;
}

__device__ real inte_B(int n, real a, real r0, real r1)
{
  real sum = 0.0;
  sum = (pow(r1, 2*n+3) - pow(r0,2*n+3))/(2*n+3)/pow(a,n) - 0.5*pow(a, n+1)*(r1*r1 - r0*r0);
  return sum;
}

__device__ real inte_M(int n, real r0, real r1)
{
  real sum = 0.0;
  sum = (pow(r1, n+3) - pow(r0, n+3))/(n+3);
  return sum;
}

__device__ real inte_N(int n, real r0, real r1)
{
  real sum = 0.0;
  if(n!=2){
    sum = (pow(r1,-n+2) - pow(r0,-n+2))/(-n+2);
  }
  else {
    sum = log(r1) - log(r0);
  }
  return sum;
}

__global__ void forcing_boussinesq_x(real alpha, real gx, real s_init, real *s, real *fx, dom_struct *dom)
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;

  for(int i = dom->Gfx._isb; i < dom->Gfx._ieb; i++) {
    if(tj < dom->Gfx._jnb && tk < dom->Gfx._knb) {
      fx[i + tj*dom->Gfx._s1b + tk*dom->Gfx._s2b]
        // buoyancy force is the opposite direction of the gravity force
        += - gx * alpha * (s[i + tj*dom->Gfx._s1b + tk*dom->Gfx._s2b] - s_init);
    }
  }
}

__global__ void forcing_boussinesq_y(real alpha, real gy, real s_init, real *s, real *fy, dom_struct *dom)
{
  int tk = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = blockIdx.y * blockDim.y + threadIdx.y;

  for(int j = dom->Gfy._jsb; j < dom->Gfy._jeb; j++) {
    if(tk < dom->Gfy._knb && ti < dom->Gfy._inb) {
      fy[ti + j*dom->Gfy._s1b + tk*dom->Gfy._s2b]
        += - gy * alpha * (s[ti + j*dom->Gfy._s1b + tk*dom->Gfy._s2b] - s_init);
    }
  }
}

__global__ void forcing_boussinesq_z(real alpha, real gz, real s_init, real *s, real *fz, dom_struct *dom)
{
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int tj = blockIdx.y * blockDim.y + threadIdx.y;

  for(int k = dom->Gfz._ksb; k < dom->Gfz._keb; k++) {
    if(ti < dom->Gfz._inb && tj < dom->Gfz._jnb) {
      fz[ti + tj*dom->Gfz._s1b + k*dom->Gfz._s2b]
        += - gz * alpha * (s[ti + tj*dom->Gfz._s1b + k*dom->Gfz._s2b] - s_init);
    }
  }
}

// this function helps to calculate the integral of heat flux on the particle surface instanteneously, it uses the lamb coefficients from scalar field and lebsque nodes
__global__ void part_heat_flux(part_struct *parts, part_struct_scalar *parts_s, real *node_t, real *node_p, real *anm_re, real *anm_im, real *anm_re00, real *anm_im00, real *anm_re_perturb, real *anm_im_perturb, int nnodes, int stride, real A1, real A2, real A3, real perturbation, real dt, real s_D)
{
  int node = threadIdx.x;
  int part = blockIdx.x;

  int i;
  // for each coefficients(n,m), calculate the surface integral; then add them together
  if(node < nnodes) {
    // for each nodes, sum the coefficients    
    real theta = node_t[node];
    real phi = node_p[node];
    parts_s[part].dsdr[node] = X_an(0, theta, phi, anm_re, anm_im, part, stride) / parts[part].r;
    for(int n = 1; n <= parts_s[part].order; n++) {
      parts_s[part].dsdr[node] += (2*n+1) * X_an(n, theta, phi, anm_re, anm_im, part, stride) / parts[part].r;
    }
    if(perturbation > 0) {
      for(int n = 0; n <= parts_s[part].order; n++) {
        real a = parts[part].r;
        real rs = parts_s[part].rs * parts[part].r;
        real denominator = pow(a, n-1)/(pow(a, 2*n+1) - pow(rs, 2*n+1));
        real tmp1 = (inte_M(n,rs,a) + pow(a, 2*n+1)*inte_N(n,a,rs)) * X_an(n, theta, phi, anm_re_perturb, anm_im_perturb, part, stride);
        real tmp2 = (inte_B(n,a,rs,a) + pow(a, 2*n+1)*inte_A(n,a,a,rs)) * perturbation_X_an(n, theta, phi, anm_re, anm_re00, anm_im, anm_im00, part, stride, dt)/s_D;
        real perturb = (tmp1 + tmp2) * denominator;
        //real perturb = (inte_B(n,a,rs,a) + pow(rs,2*n+1)*inte_A(n,a,a,rs))*pow(a, n-1)/(pow(a, 2*n+1) - pow(rs, 2*n+1)) * perturbation_X_an(n, theta, phi, anm_re, anm_re00, anm_im, anm_im00, part, stride, dt) / s_D; //without the first piece, change of particle surface temperature
        //if(node == 0) printf("pertubation correction to gradient is %f\n", perturb);
        parts_s[part].dsdr[node] += perturb;
      }
    }
    //parts_s[part].dsdr[node] = 1.0;// for test, see sphere area
    //printf("parts_s[part].dsdt[%d] is %f\n", node, parts_s[part].dsdr[node]);
    __syncthreads();


    if(node == 0) {
      parts_s[part].q = parts_s[part].dsdr[node] * A1;
      for(i = 1; i < 6; i++) {
        parts_s[part].q += A1 * parts_s[part].dsdr[node + i];
      }
      for(i = 6; i < 18; i++) {
        parts_s[part].q += A2 * parts_s[part].dsdr[node + i];
      }
      for(i = 18; i < 26; i++) {
        parts_s[part].q += A3 * parts_s[part].dsdr[node + i];
      }
      parts_s[part].q *= parts[part].r * parts[part].r;
      //printf("parts_s[part].q is, gradient is %f %f\n", parts_s[part].q, parts_s[part].dsdr[node]/parts_s[part].s);
    }
  }
}

__global__ void update_part_scalar(int nparts, part_struct *parts, part_struct_scalar *parts_s, real time, real dt, real s_k)
{
  int part = blockIdx.x;
  real vol = 4./3. * PI * parts[part].r*parts[part].r*parts[part].r;
  real m = vol * parts[part].rho;
  parts_s[part].s0 = parts_s[part].s;
  parts_s[part].s = parts_s[part].s0 + parts_s[part].update * 1.0 * parts_s[part].q * s_k * dt / m /parts_s[part].cp;
  //printf("previous, current temperature is %f %f\n", parts_s[part].s0, parts_s[part].s);
/*  if(parts_s[part].s < 1000) {
    parts_s[part].s0 = 100.0*cos(1.*(time-dt));
    parts_s[part].s = 100.0*cos(1.*time);
  } else {
    parts_s[part].s0 = 100.0;
    parts_s[part].s = 100.0;
  }
*/ 
}
