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
#include "point.h"
}

__global__ void move_points(point_struct *points, int npoints, real *u, real *v, real *w, dom_struct *dom, real mu, real dt);
/*
 *	Update point particle position after solving the fluid domain
 *	* mu fluid viscousity
 *******
 */	

#endif
