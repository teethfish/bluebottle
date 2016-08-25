

#ifndef _CUDA_SCALAR_H
#define _CUDA_SCALAR_H


extern "C"
{
#include "bluebottle.h"
#include "scalar.h"
}

#include "cuda_quadrature.h"
#include "cuda_particle.h"

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
__global__ void scalar_explicit(real *s0, real *s, real *conv_s, real *diff_s, real *conv0_s, real *diff0_s, real *u, real *v, real *w, real s_D, dom_struct *dom, real dt, real dt0);


// update scalar filed
__global__ void update_scalar(real *s, real *s0, real *conv_s, real *conv0_s, real *diff_s, real *diff0_s, dom_struct *dom);


//check if the Lebsque nodes inter-section with the wall
__global__ void check_nodes_scalar(int nparts, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, real *theta, real *phi, int nnodes, BC_s bc_s);
/*
 * Function
 * nparts: number of particle
 * parts: use the particle position information. NOTES:DOUBLE USE PARTS.NODES INFORMATION, SINCE USES IT AFTER VELOCITY FIELD,SHOULD BE OK
 * parts_s: use integrate distance information
 * dom: domain information
 * theta: 
 * phi
 * nnodes: number of nodes
 * bc_s: use the boundary condition for scalar field information
 */

// interpolate scalar field to Lebsque nodes
__global__ void interpolate_nodes_scalar(real *s, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, real *theta, real *phi, int nnodes, real *ss, BC_s bc_s);

// get the coefficients for scalar lamb's solution
__global__ void cuda_get_coeffs_scalar(part_struct *parts, part_struct_scalar *parts_s, int *nn, int *mm, real *node_t, real *node_p, real *ss, int stride_scalar, real *anm_re, real *anm_re0, real *anm_im, real *anm_im0, real *int_scalar_re, real *int_scalar_im, int nnodes, real A1, real A2, real A3, real B);
/*
 * Function
 * nn: the coefficients table for order n
 * mm: the coefficients table for order m
 * note_t: the theta of lebsque nodes
 * node_p: the phi of lebsque nodes
 * ss: the interpolated value of scalar field s
 * stride_scalar: the stride for lamb's coefficents == sum(2n+1)
 * anm_re: the real part of coefficients Anm
 * anm_re0: previous step results of coefficients Anm
 * anm_im: the imaginary part of coeffients Anm
 * anm_im0: previous step results of coefficents Anm
 * int_scalar_re: real part of integrant of (T-T_0)*Ynm at each nodes
 * int_scalar_im: imaginary part of intgrant of (T-T_0)*Ynm
 * nnodes: number of lebsque nodes
 * A1, A2, A3, B weights of spherical integrant
 */


// get the perturbation coefficients for scalar lamb's solution
__global__ void cuda_get_coeffs_scalar_perturbation(part_struct_scalar *parts_s, int *nn, int *mm, real *node_t, real *node_p, int stride_scalar, real *anm_re_perturb, real *anm_im_perturb, real *int_scalar_re, real *int_scalar_im, int nnodes, real A1, real A2, real A3, real dt, real s_D);
/*
 * Function
 * nn: the coefficients table for order n
 * mm: the coefficients table for order m
 * note_t: the theta of lebsque nodes
 * node_p: the phi of lebsque nodes
 * stride_scalar: the stride for lamb's coefficents == sum(2n+1)
 * anm_re_perturb: the real part of perturbed coefficients Anm (1/D*dTdt,Ynm) 
 * anm_im_perturb: the imaginary part of perturbed coefficients Anm (1/D*dTdt,Ynm)
 * int_scalar_re: real part of integrant of dTdt*Ynm at each nodes
 * int_scalar_im: imaginary part of intgrant of dTdt*Ynm
 * nnodes: number of lebsque nodes
 * A1, A2, A3 weights of spherical integrant
 * dt: used to calculate the time derivative of dTdt
 * s_D: thermal difisivity, coefficients are (1/D*dTdt,Ynm)
 */


__global__ void compute_error_scalar(real lamb_cut, int stride, int nparts, real *anm_re, real *anm_re0, real *anm_im, real *anm_im0, real *coeffs, real *errors, real *part_errors, dom_struct *dom);
/*
 * Function
 * lamb_cut: the maximum residue for lamb coefficients
 * stride: the stride for lamb's coefficents == sum(2n+1)
 * nparts: number of particle
 * anm_re: the real part of coefficients Anm
 * anm_im: the imaginary part of coeffients Anm
 * anm_re0: previous step results of coefficients Anm
 * anm_im0: previous step results of coefficents Anm
 * coeffs: sorted coefficents of Anm
 * errors: sorted errors
 * part_errors: store the error for each particle
 * dom: dom information
 */

__global__ void part_BC_scalar(real *s, int *phase, int *phase_shell, part_struct *parts, part_struct_scalar *parts_s, dom_struct *dom, int stride, real *anm_re, real *anm_im, real *anm_re00, real *anm_im00, real *anm_re_perturb, real *anm_im_perturb, real s_D, real perturbation, real dt);
/*
 * FUNCTION
 * s: scalar field
 * phase
 * phase_shell: the outmost cell inside the particle, aapplying the lamb'solution
 * stride: the stride for lamb's coefficents == sum(2n+1)
 * anm_re: the real part of coefficients Anm
 * anm_im: the imaginary part of coeffients Anm
 */

__device__ real X_an(int n, real theta, real phi, real *anm_re, real *anm_im, int pp, int stride);
/*
 * Function
 * n: order index of Ynm
 * theta
 * phi
 * anm_re: the real part of coefficients Anm
 * anm_im: the imaginary part of coeffients Anm
 * pp: the particle number index, the pth particle
 * stride: the stride for lamb's coefficents == sum(2n+1)
 */

/*
 * Function
 * calculate part of the perturbation correction(add time dependent time)
 */
__device__ real perturbation_X_an(int n, real theta, real phi, real *anm_re, real *anm_re00, real *anm_im, real *anm_im00, int pp, int stride, real dt);
/*
 * calculates the (dAnmdt, Ynm) part
 */

__device__ real inte_A(int n, real a, real r0, real r1);
/*
 * n: order index of Ynm
 * a: particle radius
 * r0: integration lower limit
 * r1: integration upper limit
 */

__device__ real inte_B(int n, real a, real r0, real r1);
/*
 * n: order index of Ynm
 * a: particle radius
 * r0: integration lower limit
 * r1: integration upper limit
 */


__device__ real inte_M(int n, real r0, real r1);
/*
 * calculate r^(n+3)/(n+3)
 r0: lower limit
 r1: upper limit
 */

__device__ real inte_N(int n, real r0, real r1);
/*
 * calculate r^(-n+2)/(-n+2)
 */

__global__ void show_variable(real *s0, real *s, dom_struct *dom);


__global__ void forcing_boussinesq_x(real alpha, real gx, real s_init, real *s, real *fx, dom_struct *dom);

__global__ void forcing_boussinesq_y(real alpha, real gy, real s_init, real *s, real *fy, dom_struct *dom);

__global__ void forcing_boussinesq_z(real alpha, real gz, real s_init, real *s, real *fz, dom_struct *dom);

__global__ void part_heat_flux(part_struct *parts, part_struct_scalar *parts_s, real *node_t, real *node_p, real *anm_re, real *anm_im, real *anm_re00, real *anm_im00, int nnodes, int stride, real A1, real A2, real A3, real perturbation, real dt, real s_D);

__global__ void update_part_scalar(int nparts, part_struct_scalar *parts_s, real time, real dt);

#endif



