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

#include "point.h"

int npoints;
point_struct *points;
point_struct **_points;

void points_read_input(void)
{
  int fret = 0;
  fret = fret; // prevent compiler warning	
	
  char fname[FILE_NAME_SIZE] = "";
  sprintf(fname, "%s/input/point.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  
  // read particle list
  fret = fscanf(infile, "n %d\n", &npoints);
  points = (point_struct*) malloc(npoints * sizeof(point_struct));
  cpumem += npoints * sizeof(point_struct);
  
  fclose(infile);

  printf("npoints is %d\n", npoints);
}    	  
	  

int points_init(void)
{

  for(int i = 0; i < npoints; i++) {
		points[i].r = 0.01;
    points[i].x = (2.*rng_dbl() - 1.0)*Dom.xl;
    points[i].y = (2.*rng_dbl() - 1.0)*Dom.yl;
    points[i].z = (2.*rng_dbl() - 1.0)*Dom.zl;

    points[i].u = 0.;
    points[i].v = 0.;
    points[i].w = 0.;

    points[i].ox = 0.;
    points[i].oy = 0.;
    points[i].oz = 0.;

		points[i].Fx = 0.;
    points[i].Fy = 0.;
    points[i].Fz = 0.;

    points[i].rho = 2.0;
  }
  return EXIT_SUCCESS;
}


void points_clean(void)
{
  free(points);
}
