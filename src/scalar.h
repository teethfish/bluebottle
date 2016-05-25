
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


void cuda_scalar_malloc(void);
/*
 * Function
 * if scalar_on == 1, allocate and init variables
 */

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



#endif
