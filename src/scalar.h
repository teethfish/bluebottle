
#ifndef _SCALAR_H
#define _SCALAR_H

#include "bluebottle.h"

typedef struct BC_s {
  int sW;
  real sWD;
  real sWN;
  int sE;
  real sED;
  real sEN;
  int sN;
  real sND;
  real sNN;
  int sS;
  real sSD;
  real sSN;
  int sT;
  real sTD;
  real sTN;
  int sB;
  real sBD;
  real sBN;
} BC_s;
/*
 * PURPOSE
 * MEMBERS
 * * tW -- the boundary condition type
 * * tWD -- the DIRICHLET boundary conditon value
 * * tWN -- the NEUMANN boundary condition value
 */

extern BC_s bc_s;

typedef struct part_struct_scalar {
  real s0;
  real s;
  real k;
  real rs;
  int order;
  int ncoeff;
} part_struct_scalar;
/*
 * PURPOSE
 * MEMBERS
 * * s0 is the previous time step scalar value
 * * s is the current time step scalar value
 * * order is the order to keep lamb solution, equals to index n in Ynm
 * * ncoeff is the corresponding m index in Ynm
 * * rs the integrate surface
*/

extern part_struct_scalar *parts_s;

extern part_struct_scalar **_parts_s;

extern int coeff_stride_scalar;
/*
 * stores the maximum order for all particles for lamb solution, equal to max index n in Ynm
 */

extern real lamb_cut_scalar;

extern real s_init; //initial temperature

extern real s_alpha; //coefficient of thermal expansion

extern real s_d;

extern real s_k;

extern int scalar_on;


extern real *s0;
extern real *s;
extern real *conv0_s;
extern real *conv_s;
extern real *diff0_s;
extern real *diff_s;

extern real **_s0;
extern real **_s;
extern real **_conv0_s;
extern real **_conv_s;
extern real **_diff0_s;
extern real **_diff_s;


extern real *anm_re;
extern real *anm_im;
extern real *anm_re0;
extern real *anm_im0;

extern real **_anm_re;
extern real **_anm_im;
extern real **_anm_re0;
extern real **_anm_im0;

extern int *_nn_scalar;
extern int *_mm_scalar;

void scalar_read_input(void);
/*
 * FUNCTION
 * read the scalar.config file
 ****
 */

void show_scalar_config(void);
/*
 * FUNCTION
 * show the scalar.config file
 ****
 */

void scalar_init(void);
/*
 * FUNCTION
 * allocate and init the variable
 ****
 */

void scalar_clean(void);
/*
 * FUNCTION
 * clean scalar field on host
 ****
 */

void scalar_out_restart(void);
/*
 * FUNCTION
 * write restart file for scalar field
 ***
 */

void scalar_in_restart(void);
/*
 * FUNCTION
 * read in restart file for scalar field
 ***
 */

void parts_read_input_scalar(void);
/*
 * function
 * read in particle initial scalar value, k, intergrate surface and lamb solution order
 */

void parts_init_scalar(void);
/*
 * function
 * initialize the particle scalar field value
 */

void parts_scalar_clean(void);
/*
 * FUNCTION
 * clean variables
 */

/*************************FUNCTION IN CUDA_SCALAR.CU********************/

void cuda_part_scalar_malloc(void);
/*
 * Function
 * if scalar_on == 1, allocate and init variables
 */

void cuda_part_scalar_push(void);

void cuda_part_scalar_pull(void);

void cuda_part_scalar_free(void);


void cuda_scalar_malloc(void);

void cuda_scalar_push(void);

void cuda_scalar_pull(void);

void cuda_scalar_free(void);


void cuda_scalar_BC(void);
/*
 * function
 * applying the boundary condition for scalar field before calulating
 */

void cuda_solve_scalar_explicit(void);

void cuda_update_scalar(void);

void cuda_quad_check_nodes_scalar(int dev,real *node_t, real *node_p, int nnodes);
/*
 * Function
 * check if the intergrat nodes inter-section with the wall
 */


void cuda_quad_interp_scalar(int dev, real *node_t, real *node_p, int nnodes, real *ss);
/*
 * Function
 * interpolate scalar field value into lebsque nodes
 * node_t is the theta for lebsque nodes
 * node_p is the phi for lebsque nodes
 ** nnodes is the number of lebsque nodes
 */

void cuda_scalar_lamb(void);
/*
 * FUNCTION
 * interpolate the outer field to Lebsque nodes and use the value to calculate the coefficents
 */


real cuda_scalar_lamb_err(void);
/*
 * FUNCTION
 * find the residue for lamb coefficents
 */

void cuda_part_BC_scalar(void);
/*
 * FUNCTION
 * Apply the Dirichlet boundary condition to those nodes for scalar field
 */


void cuda_show_variable(void);


void cuda_compute_boussinesq(void);


#endif
