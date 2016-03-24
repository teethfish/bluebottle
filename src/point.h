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

/****h* Bluebottle/point
 * NAME
 *  point
 * FUNCTION
 *  Bluebottle point particle functions.
 ******
 */

#ifndef _POINT_H
#define _POINT_H

#include "bluebottle.h"

/****s* point_particle/point part_struct
 * NAME
 *  point_struct
 * TYPE
 */
typedef struct point_struct {
	real r;
  real x;
  real y;
  real z;
  real u;
  real v;
  real w;
  real ox;
  real oy;
  real oz;
	real Fx;
	real Fy;
	real Fz;
  real rho;
} point_struct;
/*
 * PURPOSE
 *  Carry physical information regarding a particle.
 * MEMBERS
 *	* r -- the particle radiu
 *  * x -- the particle location component
 *  * y -- the particle location component
 *  * z -- the particle location component
 *  * u -- linear velocity in x-direction
 *  * v -- linear velocity in y-direction
 *  * w -- linear velocity in z-direction
 *  * ox -- angular velocity in x-direction
 *  * oy -- angular velocity in y-direction
 *  * oz -- angular velocity in z-direction
 *  * m -- mass
 ******
 */

/****v* particle/nparts
 * NAME
 *  nparts
 * TYPE
 */
extern int npoints;
/*
 * PURPOSE
 *  Define the total number of point particles in the domain.
 ******
 */

/****v* particle/parts
 * NAME
 *  parts
 * TYPE
 */
extern point_struct *points;
/*
 * PURPOSE
 *  A list of all particles.
 ******
 */

/****v* particle/_parts
 * NAME
 *  _parts
 * TYPE
 */
extern point_struct **_points;
/*
 * PURPOSE
 *  CUDA device analog for parts.  It contains pointers to arrays containing
 *  the particles in domain on which each device operates.  NOTE: for now,
 *  particles are designed to function on only one device.
 ******
 */

/****f* particle/parts_init()
 * NAME
 *  points_init()
 * USAGE
 *
 */
int points_init(void);
/*
 * FUNCTION
 *  Initialize the particles (build initial cages) and phase.
 * RESULT
 *  EXIT_SUCCESS if successful, EXIT_FAILURE otherwise.
 ******
 */


/****f* particle/parts_clean()
 * NAME
 *  points_clean()
 * USAGE
 */
void points_clean(void);
/*
 * FUNCTION
 *  Clean up.  Free any allocated host memory.
 ******
 */
#endif
