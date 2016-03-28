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


__global__ void move_points(dom_struct *dom, point_struct *points, g_struct g, real *conv_u, real *conv_v, real *conv_w, real *u, real *v, real *w, real *u0, real *v0, real *w0, real rho_f, real mu, real dt)
{
	int pp = threadIdx.x + blockIdx.x*blockDim.x; //point particle number

	real ddx = 1. / dom->dx;
  real ddy = 1. / dom->dy;
  real ddz = 1. / dom->dz;
	int i, j, k;
	int C;
	real m, vol;
  real xx, yy, zz;

	// each particle with a mass
  vol = 4./3. * PI * points[pp].r * points[pp].r * points[pp].r;
	m = vol * points[pp].rho;
	// interpolate u-velocity
	i = round((points[pp].x - dom->xs) * ddx) + DOM_BUF;
	j = floor((points[pp].y - dom->ys) * ddy) + DOM_BUF;
	k = floor((points[pp].z - dom->zs) * ddz) + DOM_BUF;
	xx = (i-DOM_BUF) * dom->dx + dom->xs;
  yy = (j-0.5) * dom->dy + dom->ys;
  zz = (k-0.5) * dom->dz + dom->zs;
	C = i + j*dom->Gfx.s1b + k*dom->Gfx.s2b;

  // interpolate velocity for current time step
  real dudx = 0.5*(u[C+1] - u[C-1]) * ddx;
  real dudy = 0.5*(u[C+dom->Gfx.s1b] - u[C-dom->Gfx.s1b]) * ddy;
  real dudz = 0.5*(u[C+dom->Gfx.s2b] - u[C-dom->Gfx.s2b]) * ddz;
	real uu = u[C] + dudx * (points[pp].x - xx) + dudy * (points[pp].y - yy) + dudz * (points[pp].z - zz);
  
  // interpolate velocity for previous time step
  real dudx0 = 0.5*(u0[C+1] - u0[C-1]) * ddx;
  real dudy0 = 0.5*(u0[C+dom->Gfx.s1b] - u0[C-dom->Gfx.s1b]) * ddy;
  real dudz0 = 0.5*(u0[C+dom->Gfx.s2b] - u0[C-dom->Gfx.s2b]) * ddz;
  real uu0 = u0[C] + dudx0 * (points[pp].x - xx) + dudy0 * (points[pp].y - yy) + dudz0 * (points[pp].z - zz);  

  // interpolate convection term for current time step
  real dconv_udx = 0.5*(conv_u[C+1] - conv_u[C-1]) * ddx;
  real dconv_udy = 0.5*(conv_u[C+dom->Gfx.s1b] - conv_u[C-dom->Gfx.s1b]) * ddy;
  real dconv_udz = 0.5*(conv_u[C+dom->Gfx.s2b] - conv_u[C-dom->Gfx.s2b]) * ddz;
  real conv_uu = conv_u[C] + dconv_udx * (points[pp].x - xx) + dconv_udy * (points[pp].y - yy) + dconv_udz * (points[pp].z - zz);
  real dudt = (uu - uu0)/dt + conv_uu;

  // calculate forces on the particles except the added mass term
  points[pp].Fcx = rho_f * vol * (dudt - g.x);

  // calculate the rhs of equation: the last term accounts for added mass term
  real fx = m * g.x + points[pp].Fcx + 0.5*vol*rho_f*dudt + (m + 0.5*vol*rho_f)*points[pp].u/dt;
	
  // update point particle u velocity
  points[pp].u = fx / ((m + 0.5*vol*rho_f)/dt + 6*PI*mu*points[pp].r);

  // calculate added mass force and total force
  points[pp].Fdx = 6*PI*mu*points[pp].r*(uu - points[pp].u);

  // update point particle position
  points[pp].x += points[pp].u * dt;
  while(points[pp].x > dom->xe) {
       points[pp].x -= dom->xl;
  }
  while(points[pp].x < dom->xs) {
    points[pp].x += dom->xl;
  }

	// interpolate v-velocity
  i = floor((points[pp].x - dom->xs) * ddx) + DOM_BUF;
  j = round((points[pp].y - dom->ys) * ddy) + DOM_BUF;
  k = floor((points[pp].z - dom->zs) * ddz) + DOM_BUF;
  xx = (i-0.5) * dom->dx + dom->xs;
  yy = (j-DOM_BUF) * dom->dy + dom->ys;
  zz = (k-0.5) * dom->dz + dom->zs;
	C = i + j*dom->Gfy.s1b + k*dom->Gfy.s2b;

  // interpolate v velocity for current time step
  real dvdx = 0.5*(v[C+1] - v[C-1]) * ddx;
  real dvdy = 0.5*(v[C+dom->Gfy.s1b] - v[C-dom->Gfy.s1b]) * ddy;
  real dvdz = 0.5*(v[C+dom->Gfy.s2b] - v[C-dom->Gfy.s2b]) * ddz;
 	real vv = v[C] + dvdx * (points[pp].x - xx) + dvdy * (points[pp].y - yy) + dvdz * (points[pp].z - zz);

  // interpolate v velocity for previours time step
  real dvdx0 = 0.5*(v0[C+1] - v0[C-1]) * ddx;
  real dvdy0 = 0.5*(v0[C+dom->Gfy.s1b] - v0[C-dom->Gfy.s1b]) * ddy;
  real dvdz0 = 0.5*(v0[C+dom->Gfy.s2b] - v0[C-dom->Gfy.s2b]) * ddz; 
  real vv0 = v[C] + dvdx0 * (points[pp].x - xx) + dvdy0 * (points[pp].y - yy) + dvdz0 * (points[pp].z - zz);

  // interpolate convection term for current time step
  real dconv_vdx = 0.5*(conv_v[C+1] - conv_v[C-1]) * ddx;
  real dconv_vdy = 0.5*(conv_v[C+dom->Gfy.s1b] - conv_v[C-dom->Gfy.s1b]) * ddy;
  real dconv_vdz = 0.5*(conv_v[C+dom->Gfy.s2b] - conv_v[C-dom->Gfy.s2b]) * ddz;
  real conv_vv = conv_v[C] + dconv_vdx * (points[pp].x - xx) + dconv_vdy * (points[pp].y - yy) + dconv_vdz * (points[pp].z - zz);
  real dvdt = (vv - vv0)/dt + conv_vv;

  // calculate forces on the particle in y direction except the added mass term
  points[pp].Fcy = rho_f * vol * (dvdt - g.y);

  // calculate the rhs of equation: the last term comes from added mass term
  real fy = m * g.y + points[pp].Fcy + 0.5*vol*rho_f*dvdt + (m + 0.5*vol*rho_f)*points[pp].v/dt;;

  // update point particle v velocity
  points[pp].v = fy / ((m + 0.5*vol*rho_f)/dt + 6*PI*mu*points[pp].r);

  // calculate added mass force and total force
  points[pp].Fdy = 6*PI*mu*points[pp].r*(vv - points[pp].v);

  // update point particle position in y direction
  points[pp].y += points[pp].v * dt;
  while(points[pp].y > dom->ye) {
    points[pp].y -= dom->yl;
  }
  while(points[pp].y < dom->ys) {
    points[pp].y += dom->yl;
  }


	// interpolate w-velocity
  i = floor((points[pp].x - dom->xs) * ddx) + DOM_BUF;
  j = floor((points[pp].y - dom->ys) * ddy) + DOM_BUF;
  k = round((points[pp].z - dom->zs) * ddz) + DOM_BUF;
  xx = (i-0.5) * dom->dx + dom->xs;
  yy = (j-0.5) * dom->dy + dom->ys;
  zz = (k-DOM_BUF) * dom->dz + dom->zs;
  C = i + j*dom->Gfz.s1b + k*dom->Gfz.s2b;

  // interpolate w velocity in current time step
  real dwdx = 0.5*(w[C+1] - w[C-1]) * ddx;
  real dwdy = 0.5*(w[C+dom->Gfz.s1b] - w[C-dom->Gfz.s1b]) * ddy;
  real dwdz = 0.5*(w[C+dom->Gfz.s2b] - w[C-dom->Gfz.s2b]) * ddz;
  real ww = w[C] + dwdx * (points[pp].x - xx) + dwdy * (points[pp].y - yy) + dwdz * (points[pp].z - zz);

  // interpolate w velocity in previous time step
  real dwdx0 = 0.5*(w0[C+1] - w0[C-1]) * ddx;
  real dwdy0 = 0.5*(w0[C+dom->Gfz.s1b] - w0[C-dom->Gfz.s1b]) * ddy;
  real dwdz0 = 0.5*(w0[C+dom->Gfz.s2b] - w0[C-dom->Gfz.s2b]) * ddz;
  real ww0 = w0[C] + dwdx0 * (points[pp].x - xx) + dwdy0 * (points[pp].y - yy) + dwdz0 * (points[pp].z - zz);

  // interpolate convection term in current time step
  real dconv_wdx = 0.5*(conv_w[C+1] - conv_w[C-1]) * ddx;
  real dconv_wdy = 0.5*(conv_w[C+dom->Gfz.s1b] - conv_w[C-dom->Gfz.s1b]) * ddy;
  real dconv_wdz = 0.5*(conv_w[C+dom->Gfz.s2b] - conv_w[C-dom->Gfz.s2b]) * ddz;
  real conv_ww = conv_w[C] + dconv_wdx * (points[pp].x - xx) + dconv_wdy * (points[pp].y - yy) + dconv_wdz * (points[pp].z - zz);
  real dwdt = (ww - ww0)/dt + conv_ww;
  
  // calculate forces on the particles except the added mass term
  points[pp].Fcz = rho_f * vol * (dwdt - g.z);
  
  // calculate the rhs of equation: the last term accounts for added mass term
  real fz = m * g.z + points[pp].Fcz + 0.5*vol*rho_f*dwdt + (m + 0.5*vol*rho_f)*points[pp].w/dt;
  
  // update point particle w velocity
  points[pp].w = fz / ((m + 0.5*vol*rho_f)/dt + 6*PI*mu*points[pp].r);

  // calculate added mass force and total force
  points[pp].Fdz = 6 * PI * mu * points[pp].r * (ww - points[pp].w);

	// update point particle z position
  points[pp].z += points[pp].w * dt;
  while(points[pp].z > dom->ze) {
    points[pp].z -= dom->zl;
  }
  while(points[pp].z < dom->zs) {
    points[pp].z += dom->zl;
  }

}


