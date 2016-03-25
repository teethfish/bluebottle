/*******************************************************************************
 ********************************* BLUEBOTTLE **********************************
 *******************************************************************************
 *
 *  Copyright 2012 - 2016 Adam Sierakowski, The Johns Hopkins University
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

#ifndef _CUDA_POINT_H
#define _CUDA_POINT_H

extern "C"
{
#include "bluebottle.h"
#include "domain.h"
#include "point.h"
}
__global__ void move_points(dom_struct *dom, point_struct *points, g_struct g, real *conv_u, real *conv_v, real *conv_w,real *u, real *v, real *w, real *u0, real *v0, real *w0, real rho_f, real mu, real dt);
/*
 *	Update point particle position after solving the fluid domain
 *	* dom -- dom information
 *  * points -- point particle information
 *  * g -- gravity
 *  * u -- correct flow field x velocity
 *  * v -- correct flow field y velocity
 *  * w -- correct flow field z velocity
 *  * u0 -- previous flow field x velocity
 *  * v0 -- previous flow field y velocity
 *  * w0 -- previous flow field z velocity
 *  * conv_u -- current convection term
 *  * conv_v -- current convection term
 *  * conv_w -- current convection term
 *  * rho_f -- fluid density
 *  * mu -- fluid viscosity
 *  * dt -- current time step
 *******
 */	

#endif
