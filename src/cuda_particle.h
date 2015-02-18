/*******************************************************************************
 ******************************* BLUEBOTTLE-1.0 ********************************
 *******************************************************************************
 *
 *  Copyright 2012 - 2014 Adam Sierakowski, The Johns Hopkins University
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

/****h* Bluebottle/cuda_particle_kernel
 * NAME
 *  cuda_bluebottle_kernel
 * FUNCTION
 *  Bluebottle CUDA particle kernel functions.
 ******
 */

#ifndef _CUDA_PARTICLE_H
#define _CUDA_PARTICLE_H

extern "C"
{
#include "bluebottle.h"
#include "particle.h"
}

/****f* cuda_particle_kernel/reset_phase<<<>>>()
 * NAME
 *  reset_phase<<<>>>()
 * USAGE
 */
__global__ void reset_phase(int *phase, dom_struct *dom);
/*
 * FUNCTION
 *  Set all phase array nodes to fluid (= -1).
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/reset_phase_shell<<<>>>()
 * NAME
 *  reset_phase_shell<<<>>>()
 * USAGE
 */
__global__ void reset_phase_shell(int *phase_shell, dom_struct *dom);
/*
 * FUNCTION
 *  Set all phase_shell array nodes to fluid (= 1).
 * ARGUMENTS
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */
 
/****f* cuda_particle_kernel/reset_flag_u<<<>>>()
 * NAME
 *  reset_flag_u<<<>>>()
 * USAGE
 */
__global__ void reset_flag_u(int *flag_u, dom_struct *dom, BC bc);
/*
 * FUNCTION
 *  Set all flag_u nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_u -- the device x-direction velocity flag array subdomain
 *  * dom -- the device subdomain
 *  * bc -- the boundary conditions
 ******
 */

/****f* cuda_particle_kernel/reset_flag_v<<<>>>()
 * NAME
 *  reset_flag_v<<<>>>()
 * USAGE
 */
__global__ void reset_flag_v(int *flag_v, dom_struct *dom, BC bc);
/*
 * FUNCTION
 *  Set all flag_v nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_v -- the device y-direction velocity flag array subdomain
 *  * dom -- the device subdomain
 *  * bc -- the boundary conditions
 ******
 */

/****f* cuda_particle_kernel/reset_flag_w<<<>>>()
 * NAME
 *  reset_flag_w<<<>>>()
 * USAGE
 */
__global__ void reset_flag_w(int *flag_w, dom_struct *dom, BC bc);
/*
 * FUNCTION
 *  Set all flag_w nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_w -- the device z-direction velocity flag array subdomain
 *  * dom -- the device subdomain
 *  * bc -- the boundary conditions
 ******
 */

/****f* cuda_particle_kernel/build_cage<<<>>>()
 * NAME
 *  build_cage<<<>>>()
 * USAGE
 */
__global__ void build_cage(int p, part_struct *parts, int *phase,
  int *phase_shell, dom_struct *dom, real Y, real Z,
  int js, int je, int ks, int ke);
/*
 * FUNCTION
 *  Build the cage for particle p.  Flag the particles in phase.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * parts -- the device particle array subdomain
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 *  * Y -- the virtual position of the particle
 *  * Z -- the virtual position of the particle
 *  * js -- start in y-direction
 *  * je -- end in y-direction
 *  * ks -- start in z-direction
 *  * ke -- end in z-direction
 ******
 */

/****f* cuda_particle_kernel/cage_phases_periodic_W<<<>>>()
 * NAME
 *  cage_phases_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_phases_periodic_W(int *phase, int *phase_shell,
  dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary phase and phase_shell for periodic conditions.
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_phases_periodic_E<<<>>>()
 * NAME
 *  cage_phases_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_phases_periodic_E(int *phase, int *phase_shell,
  dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary phase and phase_shell for periodic conditions.
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_phases_periodic_S<<<>>>()
 * NAME
 *  cage_phases_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_phases_periodic_S(int *phase, int *phase_shell,
  dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary phase and phase_shell for periodic conditions.
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_phases_periodic_N<<<>>>()
 * NAME
 *  cage_phases_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_phases_periodic_N(int *phase, int *phase_shell,
  dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary phase and phase_shell for periodic conditions.
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_phases_periodic_B<<<>>>()
 * NAME
 *  cage_phases_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_phases_periodic_B(int *phase, int *phase_shell,
  dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary phase and phase_shell for periodic conditions.
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_phases_periodic_T<<<>>>()
 * NAME
 *  cage_phases_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_phases_periodic_T(int *phase, int *phase_shell,
  dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary phase and phase_shell for periodic conditions.
 * ARGUMENTS
 *  * phase -- the device phase array subdomain
 *  * phase_shell -- the device phase_shell array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_1<<<>>>()
 * NAME
 *  cage_flag_u_1<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_1(int p, int *flag_u, part_struct *parts,
  dom_struct *dom, int *phase, int *phase_shell,
  int js, int je, int ks, int ke);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the x-direction velocity. Step 1:
 *  flag outer cage nodes in each direction.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * flag_u -- the device flag array subdomain
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * phase -- the phase array
 *  * js -- start in y-direction
 *  * je -- end in y-direction
 *  * ks -- start in z-direction
 *  * ke -- end in z-direction
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_1<<<>>>()
 * NAME
 *  cage_flag_v_1<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_1(int p, int *flag_v, part_struct *parts,
  dom_struct *dom, int *phase, int *phase_shell,
  int is, int ie, int ks, int ke);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the y-direction velocity. Step 1:
 *  flag outer cage nodes in each direction.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * flag_v -- the device flag array subdomain
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * phase -- the phase array
 *  * is -- start in x-direction
 *  * ie -- end in x-direction
 *  * ks -- start in z-direction
 *  * ke -- end in z-direction
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_1<<<>>>()
 * NAME
 *  cage_flag_w_1<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_1(int p, int *flag_w, part_struct *parts,
  dom_struct *dom, int *phase, int *phase_shell,
  int is, int ie, int js, int je);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the z-direction velocity. Step 1:
 *  flag outer cage nodes in each direction.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * flag_w -- the device flag array subdomain
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * phase -- the phase array
 *  * is -- start in x-direction
 *  * ie -- end in x-direction
 *  * js -- start in y-direction
 *  * je -- end in y-direction
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_W<<<>>>()
 * NAME
 *  cage_flag_u_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_W(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_E<<<>>>()
 * NAME
 *  cage_flag_u_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_E(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_S<<<>>>()
 * NAME
 *  cage_flag_u_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_S(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_N<<<>>>()
 * NAME
 *  cage_flag_u_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_N(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_B<<<>>>()
 * NAME
 *  cage_flag_u_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_B(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_T<<<>>>()
 * NAME
 *  cage_flag_u_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_T(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_W<<<>>>()
 * NAME
 *  cage_flag_v_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_W(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_E<<<>>>()
 * NAME
 *  cage_flag_v_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_E(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_S<<<>>>()
 * NAME
 *  cage_flag_v_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_S(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_N<<<>>>()
 * NAME
 *  cage_flag_v_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_N(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_B<<<>>>()
 * NAME
 *  cage_flag_v_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_B(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_T<<<>>>()
 * NAME
 *  cage_flag_v_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_T(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_W<<<>>>()
 * NAME
 *  cage_flag_w_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_W(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_E<<<>>>()
 * NAME
 *  cage_flag_w_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_E(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_S<<<>>>()
 * NAME
 *  cage_flag_w_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_S(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_N<<<>>>()
 * NAME
 *  cage_flag_w_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_N(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_B<<<>>>()
 * NAME
 *  cage_flag_w_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_B(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_T<<<>>>()
 * NAME
 *  cage_flag_w_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_T(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_2<<<>>>()
 * NAME
 *  cage_flag_u_2<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_2(int p, int *flag_u, int *flag_v, int *flag_w,
  part_struct *parts, dom_struct *dom, int *phase,
  int js, int je, int ks, int ke);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the x-direction velocity. Step 2:
 *  if a cage cell has flags in another component, also flag this velocity.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * flag_u -- the device flag array subdomain
 *  * flag_v -- the device flag array subdomain
 *  * flag_w -- the device flag array subdomain
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * phase -- the phase array
 *  * js -- start in y-direction
 *  * je -- end in y-direction
 *  * ks -- start in z-direction
 *  * ke -- end in z-direction
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_2<<<>>>()
 * NAME
 *  cage_flag_v_2<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_2(int p, int *flag_u, int *flag_v, int *flag_w,
  part_struct *parts, dom_struct *dom, int *phase,
  int is, int ie, int ks, int ke);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the y-direction velocity. Step 2:
 *  if a cage cell has flags in another component, also flag this velocity.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * flag_u -- the device flag array subdomain
 *  * flag_v -- the device flag array subdomain
 *  * flag_w -- the device flag array subdomain
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * phase -- the phase array
 *  * is -- start in x-direction
 *  * ie -- end in x-direction
 *  * ks -- start in z-direction
 *  * ke -- end in z-direction
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_2<<<>>>()
 * NAME
 *  cage_flag_w_2<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_2(int p, int *flag_u, int *flag_v, int *flag_w,
  part_struct *parts, dom_struct *dom, int *phase,
  int is, int ie, int js, int je);
/*
 * FUNCTION
 *  Flag the boundaries of the particle for the z-direction velocity. Step 2:
 *  if a cage cell has flags in another component, also flag this velocity.
 * ARGUMENTS
 *  * p -- the particle index on which to operate
 *  * flag_u -- the device flag array subdomain
 *  * flag_v -- the device flag array subdomain
 *  * flag_w -- the device flag array subdomain
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * phase -- the phase array
 *  * is -- start in x-direction
 *  * ie -- end in x-direction
 *  * js -- start in y-direction
 *  * je -- end in y-direction
 ******
 */

/****f* cuda_particle_kernel/part_BC_u<<<>>>()
 * NAME
 *  part_BC_u<<<>>>()
 * USAGE
 */
__global__ void part_BC_u(real *u, int *phase, int *flag_u,
  part_struct *parts, dom_struct *dom,
  real nu, int stride,
  real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  real *chinm_re, real *chinm_im);
/*
 * FUNCTION
 *  Apply u-velocity boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * i -- the particle index on which to operate
 *  * u -- the device flow velocity field
 *  * phase -- particle phase flag
 *  * flag_u -- the particle boundary flag on u-velocities
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * nu -- the fluid kinematic viscosity
 *  * stride -- stride length for Lamb's coefficient storage arrays
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * chinm_re -- the real part of Lamb's coefficient chi_{nm}
 *  * chinm_im -- the imaginary part of Lamb's coefficient chi_{nm}
 ******
 */

/****f* cuda_particle_kernel/part_BC_v<<<>>>()
 * NAME
 *  part_BC_v<<<>>>()
 * USAGE
 */
__global__ void part_BC_v(real *v, int *phase, int *flag_v,
  part_struct *parts, dom_struct *dom,
  real nu, int stride,
  real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  real *chinm_re, real *chinm_im);
/*
 * FUNCTION
 *  Apply v-velocity boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * i -- the particle index on which to operate
 *  * v -- the device flow velocity field
 *  * phase -- particle phase flag
 *  * flag_v -- the particle boundary flag on v-velocities
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * nu -- the fluid kinematic viscosity
 *  * stride -- stride length for Lamb's coefficient storage arrays
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * chinm_re -- the real part of Lamb's coefficient chi_{nm}
 *  * chinm_im -- the imaginary part of Lamb's coefficient chi_{nm}
 ******
 */

/****f* cuda_particle_kernel/part_BC_w<<<>>>()
 * NAME
 *  part_BC_w<<<>>>()
 * USAGE
 */
__global__ void part_BC_w(real *w, int *phase, int *flag_w,
  part_struct *parts, dom_struct *dom,
  real nu, int stride,
  real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  real *chinm_re, real *chinm_im);
/*
 * FUNCTION
 *  Apply w-velocity boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * i -- the particle index on which to operate
 *  * w -- the device flow velocity field
 *  * phase -- particle phase flag
 *  * flag_w -- the particle boundary flag on w-velocities
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * nu -- the fluid kinematic viscosity
 *  * stride -- stride length for Lamb's coefficient storage arrays
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * chinm_re -- the real part of Lamb's coefficient chi_{nm}
 *  * chinm_im -- the imaginary part of Lamb's coefficient chi_{nm}
 ******
 */

/****f* cuda_particle_kernel/part_BC_p<<<>>>()
 * NAME
 *  part_BC_p<<<>>>()
 * USAGE
 */
__global__ void part_BC_p(real *p, real *p_rhs, int *phase, int *phase_shell,
  part_struct *parts, dom_struct *dom,
  real mu, real nu, real dt, real dt0, gradP_struct gradP, real rho_f, int stride,
  real *pnm_re00, real *pnm_im00, real *phinm_re00, real *phinm_im00,
  real *chinm_re00, real *chinm_im00,
  real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  real *chinm_re, real *chinm_im);
/*
 * FUNCTION
 *  Apply pressure boundary condition to particle i. This routine uses the
 *  Lamb's coefficients calculated previously to determine velocity boundary
 *  conditions on the particle.
 * ARGUMENTS
 *  * p -- the device flow velocity field
 *  * p_rhs -- the Poisson problem right-hand side
 *  * phase_shell -- the particle shell boundary flag on pressure
 *  * phase -- the particle boundary flag on pressure
 *  * parts -- the device particle array subdomain
 *  * dom -- the device subdomain
 *  * mu -- the fluid dynamic viscosity
 *  * nu -- the fluid kinematic viscosity
 *  * dt -- time step size
 *  * gradP -- the body force
 *  * rho_f -- fluid density
 *  * stride -- stride length for Lamb's coefficient storage arrays
 *  * pnm_re00 -- the real part of Lamb's coefficient p_{nm} (last timestep)
 *  * pnm_im00 -- the imaginary part of Lamb's coefficient p_{nm}
        (last timestep)
 *  * phinm_re00 -- the real part of Lamb's coefficient phi_{nm}
        (last timestep)
 *  * phinm_im00 -- the imaginary part of Lamb's coefficient phi_{nm}
        (last timestep)
 *  * chinm_re00 -- the real part of Lamb's coefficient chi_{nm}
        (last timestep)
 *  * chinm_im00 -- the imaginary part of Lamb's coefficient chi_{nm}
        (last timestep)
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * chinm_re -- the real part of Lamb's coefficient chi_{nm}
 *  * chinm_im -- the imaginary part of Lamb's coefficient chi_{nm}
 ******
 */

/****f* cuda_particle/xyz2rtp<<<>>>()
 * NAME
 *  xyz2rtp<<<>>>()
 * USAGE
 */
__device__ void xyz2rtp(real x, real y, real z, real *r, real *t, real *p);
/*
 * PURPOSE
 *  Compute (r, theta, phi) from (x, y, z).
 * ARGUMENTS
 *  x -- x-component in Cartesian basis
 *  y -- y-component in Cartesian basis
 *  z -- z-component in Cartesian basis
 *  r -- r-component in spherical basis
 *  t -- theta-component in spherical basis
 *  p -- phi-component in spherical basis
 ******
 */

/****f* cuda_particle/Nnm<<<>>>()
 * NAME
 *  Nnm<<<>>>()
 * USAGE
 */
__device__ real Nnm(int n, int m);
/*
 * PURPOSE
 *  Compute spherical harmonic normalization N_nm.
 * ARGUMENTS
 *  * n -- degree
 *  * m -- order
 ******
 */

/****f* cuda_particle/Pnm<<<>>>()
 * NAME
 *  Pnm<<<>>>()
 * USAGE
 */
__device__ real Pnm(int n, int m, real t);
/*
 * PURPOSE
 *  Compute associated Legendre function P_nm(theta).
 * ARGUMENTS
 *  * n -- degree
 *  * m -- order
 *  * t -- theta
 ******
 */

/****f* cuda_particle/pn<<<>>>()
 * NAME
 *  pn<<<>>>()
 * USAGE
 */
__device__ real pn(int n, real a, real r, real theta, real phi,
  real *pnm_re, real *pnm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum pn.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/phin<<<>>>()
 * NAME
 *  phin<<<>>>()
 * USAGE
 */
__device__ real phin(int n, real a, real r, real theta, real phi,
  real *phinm_re, real *phinm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum phin.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/Dpnt<<<>>>()
 * NAME
 *  Dpnt<<<>>>()
 * USAGE
 */
__device__ real Dpnt(int n, real a, real r, real theta, real phi,
  real *pnm_re, real *pnm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum dp_n/dtheta.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/Dpnp<<<>>>()
 * NAME
 *  Dpnp<<<>>>()
 * USAGE
 */
__device__ real Dpnp(int n, real a, real r, real theta, real phi,
  real *pnm_re, real *pnm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum Dp_n/dphi.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/Dphint<<<>>>()
 * NAME
 *  Dphint<<<>>>()
 * USAGE
 */
__device__ real Dphint(int n, real a, real r, real theta, real phi,
  real *phinm_re, real *phinm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum dphi_n/dtheta.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/Dphinp<<<>>>()
 * NAME
 *  Dphinp<<<>>>()
 * USAGE
 */
__device__ real Dphinp(int n, real a, real r, real theta, real phi,
  real *phinm_re, real *phinm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum dphi_n/dphi.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/DXxchint<<<>>>()
 * NAME
 *  DXxchint<<<>>>()
 * USAGE
 */
__device__ real DXxchint(int n, real a, real r, real theta, real phi,
  real *chinm_re, real *chinm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum curl(x*chi_n) dot e_theta.
 * ARGUMENTS
 *  * TODO!
 ******
 */

/****f* cuda_particle/DXxchinp<<<>>>()
 * NAME
 *  DXxchinp<<<>>>()
 * USAGE
 */
__device__ real DXxchinp(int n, real a, real r, real theta, real phi,
  real *chinm_re, real *chinm_im, int i, int stride);
/*
 * PURPOSE
 *  Compute sum curl(x*chi_n) dot e_phi.
 * ARGUMENTS
 *  * TODO!
 ******
 */


/****f* cuda_particle_kernel/X_pn<<<>>>()
 * NAME
 *  X_pn<<<>>>()
 * USAGE
 */
__device__ real X_pn(int n, real theta, real phi,
  real *pnm_re, real *pnm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/X_phin<<<>>>()
 * NAME
 *  X_phin<<<>>>()
 * USAGE
 */
__device__ real X_phin(int n, real theta, real phi,
  real *phinm_re, real *phinm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/Y_pn<<<>>>()
 * NAME
 *  Y_pn<<<>>>()
 * USAGE
 */
__device__ real Y_pn(int n, real theta, real phi,
  real *pnm_re, real *pnm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/Y_phin<<<>>>()
 * NAME
 *  Y_phin<<<>>>()
 * USAGE
 */
__device__ real Y_phin(int n, real theta, real phi,
  real *phinm_re, real *phinm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/Y_chin<<<>>>()
 * NAME
 *  Y_chin<<<>>>()
 * USAGE
 */
__device__ real Y_chin(int n, real theta, real phi,
  real *chinm_re, real *chinm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * chinm_re -- the real part of Lamb's coefficient chi_{nm}
 *  * chinm_im -- the imaginary part of Lamb's coefficient chi_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/Z_pn<<<>>>()
 * NAME
 *  Z_pn<<<>>>()
 * USAGE
 */
__device__ real Z_pn(int n, real theta, real phi,
  real *pnm_re, real *pnm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMETNS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/Z_phin<<<>>>()
 * NAME
 *  Z_phin<<<>>>()
 * USAGE
 */
__device__ real Z_phin(int n, real theta, real phi,
  real *phinm_re, real *phinm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/Z_chin<<<>>>()
 * NAME
 *  Z_chin<<<>>>()
 * USAGE
 */
__device__ real Z_chin(int n, real theta, real phi,
  real *chinm_re, real *chinm_im, int pp, int stride);
/*
 * FUNCTION
 *  Helper function for calculating sums involved in Lamb's solution
 *  for velocity.
 * ARGUMENTS
 *  * n -- sum iterate
 *  * theta -- spherical angle
 *  * phi -- spherical angle
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * pp -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 ******
 */

/****f* cuda_particle_kernel/lamb_vel<<<>>>()
 * NAME
 *  lamb_vel<<<>>>()
 * USAGE
 */
__device__ void lamb_vel(int order, real a, real r, real theta, real phi,
  real nu, real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  real *chinm_re, real *chinm_im,
  int p_ind, int stride, real *Ux, real *Uy, real *Uz);
/*
 * FUNCTION
 *  Calculate the velocities in the Cartesian directions from the
 *  analytic solution using the Lamb's coefficients.
 * ARGUMENTS
 *  * order -- Lamb's coefficient truncation order
 *  * a -- particle radius
 *  * r -- radial position
 *  * theta -- polar angle position
 *  * phi -- azimuthal angle position
 *  * nu -- the fluid kinematic viscosity
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * chinm_re -- the real part of Lamb's coefficient chi_{nm}
 *  * chinm_im -- the imaginary part of Lamb's coefficient chi_{nm}
 *  * p_ind -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 *  * Ux -- the x-direction pressure gradient
 *  * Ux -- the y-direction pressure gradient
 *  * Ux -- the z-direction pressure gradient
 ******
 */

/****f* cuda_particle_kernel/lamb_gradP<<<>>>()
 * NAME
 *  lamb_gradP<<<>>>()
 * USAGE
 */
__device__ void lamb_gradP(int order, real a, real r, real theta, real phi,
  real mu, real nu, real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  int p_ind, int stride, real *gPx, real *gPy, real *gPz);
/*
 * FUNCTION
 *  Calculate the pressure gradients in the Cartesian directions from the
 *  analytic solution using the Lamb's coefficients.
 * ARGUMENTS
 *  * order -- Lamb's coefficient truncation order
 *  * a -- particle radius
 *  * r -- radial position
 *  * theta -- polar angle position
 *  * phi -- azimuthal angle position
 *  * mu -- the fluid dynamic viscosity
 *  * nu -- the fluid kinematic viscosity
 *  * pnm_re -- the real part of Lamb's coefficient p_{nm}
 *  * pnm_im -- the imaginary part of Lamb's coefficient p_{nm}
 *  * phinm_re -- the real part of Lamb's coefficient phi_{nm}
 *  * phinm_im -- the imaginary part of Lamb's coefficient phi_{nm}
 *  * p_ind -- Lamb's coefficient access index helper value
 *  * stride -- Lamb's coefficient array stride length
 *  * gPx -- the x-direction pressure gradient
 *  * gPy -- the y-direction pressure gradient
 *  * gPz -- the z-direction pressure gradient
 ******
 */

/****f* cuda_particle_kernel/predict_coeffs<<<>>>()
 * NAME
 *  predict_coeffs<<<>>>()
 * USAGE
 */
__global__ void predict_coeffs(real dt0, real dt,
  real *pnm_re00, real *pnm_im00, real *phinm_re00, real *phinm_im00,
  real *chinm_re00, real *chinm_im00,
  real *pnm_re0, real *pnm_im0, real *phinm_re0, real *phinm_im0,
  real *chinm_re0, real *chinm_im0,
  real *pnm_re, real *pnm_im, real *phinm_re, real *phinm_im,
  real *chinm_re, real *chinm_im, int stride);
/*
 * FUNCTION
 * ARGUMENTS
 ******
 */

#endif
