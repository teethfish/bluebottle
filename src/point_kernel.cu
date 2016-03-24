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

#include "cuda_point.h"


__global__ void move_points(point_struct *points, int npoints, real *u, real *v, real *w, dom_struct *dom, real mu, real dt)
{
	int pp = threadIdx.x + blockIdx.x*blockDim.x; //point particle number

	real ddx = 1. / dom->dx;
  real ddy = 1. / dom->dy;
  real ddz = 1. / dom->dz;
	int i, j, k;
	int C;
	real m, xx, yy, zz;

	// each particle with a mass
	m = 4./3. * PI * points[pp].r * points[pp].r * points[pp].r * points[pp].rho;

	// interpolate u-velocity
	i = round((points[pp].x - dom->xs) * ddx) + DOM_BUF;
	j = floor((points[pp].y - dom->ys) * ddy) + DOM_BUF;
	k = floor((points[pp].z - dom->zs) * ddz) + DOM_BUF;
	xx = (i-DOM_BUF) * dom->dx + dom->xs;
  yy = (j-0.5) * dom->dy + dom->ys;
  zz = (k-0.5) * dom->dz + dom->zs;
	C = i + j*dom->Gfx.s1b + k*dom->Gfx.s2b;
  real dudx = 0.5*(u[C+1] - u[C-1]) * ddx;
  real dudy = 0.5*(u[C+dom->Gfx.s1b] - u[C-dom->Gfx.s1b]) * ddy;
  real dudz = 0.5*(u[C+dom->Gfx.s2b] - u[C-dom->Gfx.s2b]) * ddz;
	real uu = u[C] + dudx * (points[pp].x - xx) + dudy * (points[pp].y - yy) + dudz * (points[pp].z - zz);
	// update point particle u velocity
	points[pp].Fx = 6*PI*mu*points[pp].r*(uu - points[pp].u);
	points[pp].u = points[pp].Fx * dt / m + points[pp].u;

	// interpolate v-velocity
  i = floor((points[pp].x - dom->xs) * ddx) + DOM_BUF;
  j = round((points[pp].y - dom->ys) * ddy) + DOM_BUF;
  k = floor((points[pp].z - dom->zs) * ddz) + DOM_BUF;
  xx = (i-0.5) * dom->dx + dom->xs;
  yy = (j-DOM_BUF) * dom->dy + dom->ys;
  zz = (k-0.5) * dom->dz + dom->zs;
	C = i + j*dom->Gfy.s1b + k*dom->Gfy.s2b;
  real dvdx = 0.5*(v[C+1] - v[C-1]) * ddx;
  real dvdy = 0.5*(v[C+dom->Gfy.s1b] - v[C-dom->Gfy.s1b]) * ddy;
  real dvdz = 0.5*(v[C+dom->Gfy.s2b] - v[C-dom->Gfy.s2b]) * ddz;
 	real vv = v[C] + dvdx * (points[pp].x - xx) + dvdy * (points[pp].y - yy) + dvdz * (points[pp].z - zz);
	// update point particle v velocity
	points[pp].Fy = 6*PI*mu*points[pp].r*(vv - points[pp].v);
	points[pp].v = points[pp].Fy * dt / m + points[pp].v;	

	// interpolate w-velocity
  i = floor((points[pp].x - dom->xs) * ddx) + DOM_BUF;
  j = floor((points[pp].y - dom->ys) * ddy) + DOM_BUF;
  k = round((points[pp].z - dom->zs) * ddz) + DOM_BUF;
  xx = (i-0.5) * dom->dx + dom->xs;
  yy = (j-0.5) * dom->dy + dom->ys;
  zz = (k-DOM_BUF) * dom->dz + dom->zs;
  C = i + j*dom->Gfz.s1b + k*dom->Gfz.s2b;
  real dwdx = 0.5*(w[C+1] - w[C-1]) * ddx;
  real dwdy = 0.5*(w[C+dom->Gfz.s1b] - w[C-dom->Gfz.s1b]) * ddy;
  real dwdz = 0.5*(w[C+dom->Gfz.s2b] - w[C-dom->Gfz.s2b]) * ddz;
  real ww = w[C] + dwdx * (points[pp].x - xx) + dwdy * (points[pp].y - yy) + dwdz * (points[pp].z - zz);
	// update point particle w velocity
	points[pp].Fz = 6*PI*mu*points[pp].r*(ww - points[pp].w);
	points[pp].w = points[pp].Fz * dt / m + points[pp].w;
}


