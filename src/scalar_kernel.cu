

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

__global__ void scalar_explicit(real *s0, real *s, real *conv_s, real *diff_s, real *u, real *v, real *w, real s_k, dom_struct *dom, real dt)
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
    
      /*if(s[C] > 90.0 || s[C] < -10.){
        //printf("s[%d] is %f, i,j,k is %d %d %d, conv,diff is %f %f; is %f %f %f %f %f %f\n", C, s[C], i, tj, tk, conv_s[C], diff_s[C],s0[Cx0],s0[Cx1],s0[Cy0],s0[Cy1], s0[Cz0], s0[Cz1]);
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
      if(s0[C] != 0 || s[C] != 0 || (C > 15625 && C < 16000)){
        printf("s0[%d,%d,%d] and s[%d] is %f %f\n", i, tj, tk, C, s0[C], s[C]);
      }
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
  //printf("dsdx, dsdy, dsdz is %f %f %f, x-xx, y-yy, z-zz is %f %f %f\n", dsdx, dsdy, dsdz, x-xx, y-yy, z-zz);
  //printf("nodes %d is %f, s[C] is %f\n", node+nnodes*part, ss[node+nnodes*part], s[C]);
  // wall temperature if this node intersects wall
  // TODO: right now set it to be 0
  ss[node+nnodes*part] = (parts[part].nodes[node]==-1)*ss[node+part*nnodes];
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

    //TODO:since now m ranges from -n to n, remember to update function of nnm,pnm
    real N_nm = nnm(n,m);
    real P_nm = pnm(n,m,theta);

    int_scalar_re[j] += N_nm*P_nm*ss[node+part*nnodes]*cos(m*phi);
    int_scalar_im[j] += -N_nm*P_nm*ss[node+part*nnodes]*sin(m*phi);
    //printf("int_scalar_re[%d] is %f\n", j, int_scalar_re[j]);

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
      //printf("anm_re[%d] is %f\n", stride_scalar*part+coeff, anm_re[stride_scalar*part+coeff]);
      //printf("anm_im[%d] is %f\n", stride_scalar*part+coeff, anm_im[stride_scalar*part+coeff]);
    }
  }
} 
        

__global__ void compute_error_scalar(real lamb_cut, int stride, int nparts, real *anm_re, real *anm_re0, real *anm_im, real *anm_im0, real *coeffs, real *errors, real *part_errors, dom_struct *dom)
{
  int part = blockIdx.x;
  int i,j;
  real tmp = FLT_MIN;
  int loc = 0;
  real avg = 0;
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

  // compute the average of the coefficients
  for(i = 0; i < stride*2; i++) {
    avg += s_coeffs[i]*s_coeffs[i];
  }
  avg = avg / (stride*2.);

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
    coeffs[part*stride+i] = s_coeffs[loc];

    // if its corresponding coefficient has large enough magnitude,
    // compute error for this coefficient
    if(fabs(s_coeffs[loc]) > lamb_cut*fabs(coeffs[part*stride+0])) {
      div = fabs(s_coeffs[loc]);// + fabs(avg)*1e-4;
      if(div < 1e-16) div = 1e-16;
      errors[part*stride+i] = fabs((s_coeffs[loc] - s_coeffs0[loc]) / div);
    } else errors[part*stride+i] = 0.;

    // discard this value since we've used it once
    s_coeffs[loc] = 0.;
  }

  // find the largest error for each particle
  tmp = FLT_MIN;
  for(i = 0; i < 2*stride; i++) {
    if(errors[part*stride+i] > tmp) tmp = errors[part*stride+i];
  }

  // write error to return for each particle
  part_errors[part] = tmp;
}

__global__ void part_BC_scalar(real *s, int *phase, int *phase_shell, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, int stride, real *anm_re, real *anm_im)
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
        /*for(int l = 0; l < 9; l++){
          printf("anm_re[%d] anm_im[%d] is %f %f\n",l,l, anm_re[l], anm_im[l]);
        }*/
        ss_tmp = (pow(ra,0.0) - pow(ar,1.)) * X_an(0, theta, phi, anm_re, anm_im, P, stride);
        for(int n = 1; n <= order; n++) {
          ss_tmp += (pow(ra,n) - pow(ar,n+1)) * X_an(n, theta, phi, anm_re, anm_im, P, stride);
        }
        ss_tmp += s_surface;
        //printf("ss_tmp is %f\n", ss_tmp);
        // only apply value at nodes inside particle & phase_shell==0
        // phase_shell = 1 means normal nodes, phase_shell = 0 means pressure nodes
        s[CC] = ss_tmp * (phase[CC] > -1 && phase_shell[CC] < 1) + (phase_shell[CC] > 0)*s_surface; 
        /*if(phase_shell[CC] < 1){
          printf("s[CC] is %f, ar is %f\n", s[CC], ar);
        }*/
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
































