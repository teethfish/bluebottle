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


  // read point particle density and radius
  if(npoints > 0)  {
#ifdef DOUBLE
    fret = fscanf(infile, "rho %lf\n", &points[0].rho);
    fret = fscanf(infile, "r %lf\n", &points[0].r);
#else
    fret = fscanf(infile, "rho %f\n", &points[0].rho);           
    fret = fscanf(infile, "r %f\n", &points[0].r);    
#endif
  }  
  fclose(infile);

  printf("npoints is %d\n", npoints);
}    	  
	  

int points_init(void)
{

  for(int i = 0; i < npoints; i++) {
		points[i].r = points[0].r;
    points[i].rho = points[0].rho;

    points[i].x = (rng_dbl() - 0.5)*Dom.xl;
    points[i].y = (rng_dbl() - 0.5)*Dom.yl;
    points[i].z = (rng_dbl() - 0.5)*Dom.zl;

    points[i].u = 0.;
    points[i].v = 0.;
    points[i].w = 0.;

    points[i].ox = 0.;
    points[i].oy = 0.;
    points[i].oz = 0.;

		points[i].Fx = 0.;
    points[i].Fy = 0.;
    points[i].Fz = 0.;
  
    points[i].Fax = 0.;
    points[i].Fay = 0.;
    points[i].Faz = 0.;

    points[i].Fdx = 0.;
    points[i].Fdy = 0.;
    points[i].Fdz = 0.;

    points[i].Fcx = 0.;
    points[i].Fcy = 0.;
    points[i].Fcz = 0.;
  }
  return EXIT_SUCCESS;
}

void points_show(void)
{
  for(int i = 0; i < npoints; i++) {
    printf("points[%d].u is %f\n", i, points[i].u);
    printf("points[%d].v is %f\n", i, points[i].v);
    printf("points[%d].w is %f\n", i, points[i].w);
    printf("points[%d].Fx is %f\n", i, points[i].Fx);
    printf("points[%d].Fy is %f\n", i, points[i].Fy);
    printf("points[%d].Fz is %f\n", i, points[i].Fz);
  }
}


void points_clean(void)
{
  free(points);
}


