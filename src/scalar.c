#include "scalar.h"

BC_s bc_s;
real s_d;
real s_k;
int scalar_on;

real *s0;
real *s;
real *conv0_s;
real *conv_s;
real *diff0_s;
real *diff_s;


real **_s0;
real **_s;
real **_conv0_s;
real **_conv_s;
real **_diff0_s;
real **_diff_s;

void scalar_read_input(void)
{

  int fret = 0;
  fret = fret; // prevent compiler warning

  cpumem = 0;
  gpumem = 0;

  // open configuration file for reading
  char fname[FILE_NAME_SIZE] = "";
  sprintf(fname, "%s/input/scalar.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    fprintf(stderr, "No scalar field will be calculated.\n");
    scalar_on = 0;
  }
  else {
    scalar_on = 1;
    char buf[CHAR_BUF_SIZE] = "";  // character read buffer
    fret = fscanf(infile, "scalar_on %d\n", &scalar_on);
    if (scalar_on == 1) {    
      // read domain
#ifdef DOUBLE
      fret = fscanf(infile, "diffusivity %lf\n", &s_d);
      fret = fscanf(infile, "conductivity %lf\n", &s_k);
#else
      fret = fscanf(infile, "diffusivity %f\n", &s_d);
      fret = fscanf(infile, "conductivity %f\n", &s_k);
#endif
      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "BOUNDARY CONDITIONS\n");
      fret = fscanf(infile, "bc_s.sW %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sW = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sW = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sWD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sW = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sWN);
      } else {
        fprintf(stderr, "flow.config read error in W boundary condition.\n");
        exit(EXIT_FAILURE);
      }

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sE %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sE = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sE = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sED);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sE = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sEN); 
      } else {
        fprintf(stderr, "flow.config read error in E boundary condition.\n");
        exit(EXIT_FAILURE);
      }      

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sN %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sN = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sN = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sND);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sN = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sNN);
      } else {
        fprintf(stderr, "flow.config read error in N boundary condition.\n");
        exit(EXIT_FAILURE);
      }

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sS %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sS = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sS = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sSD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sS = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sSN);
      } else {
        fprintf(stderr, "flow.config read error in S boundary condition.\n");
        exit(EXIT_FAILURE);
      }             

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sB %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sB = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sB = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sBD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sB = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sBN);
      } else {
        fprintf(stderr, "flow.config read error in B boundary condition.\n");
        exit(EXIT_FAILURE);
      }      

      fret = fscanf(infile, "\n");
      fret = fscanf(infile, "bc_s.sT %s", buf);
      if(strcmp(buf, "PERIODIC") == 0) {
        bc_s.sT = PERIODIC;
      } else if(strcmp(buf, "DIRICHLET") == 0) {
        bc_s.sT = DIRICHLET;
        fret = fscanf(infile, "%lf", &bc_s.sTD);
      } else if(strcmp(buf, "NEUMANN") == 0) {
        bc_s.sT = NEUMANN;
        fret = fscanf(infile, "%lf", &bc_s.sTN);
      } else {
        fprintf(stderr, "flow.config read error in T boundary condition.\n");
        exit(EXIT_FAILURE);
      }

    }
  }
}


void show_scalar_config(void)
{
  printf("Show scalar.config...\n");
  printf("scalar_on is %d\n", scalar_on);
  printf("diffusivity is %f\n", s_d);
  printf("conductivity is %f\n", s_k);
  printf("Boundary condition is:\n");
  printf("  On W ");
  if(bc_s.sW == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sWD);
  else if(bc_s.sW == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sWN);
  else if(bc_s.sW == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
  else printf(" bc_s.sW is wrong with value %d\n", bc_s.sW);
  
  printf("  On E ");
  if(bc_s.sE == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sED);
  else if(bc_s.sE == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sEN);
  else if(bc_s.sE == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
  else printf(" bc_s.sE is wrong with value %d\n", bc_s.sE);

  printf("  On N ");
  if(bc_s.sN == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sND);
  else if(bc_s.sN == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sNN);
  else if(bc_s.sN == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
  else printf(" bc_s.sN is wrong with value %d\n", bc_s.sN);

  printf("  On S ");
  if(bc_s.sS == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sSD);
  else if(bc_s.sS == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sSN);
  else if(bc_s.sS == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
  else printf(" bc_s.sS is wrong with value %d\n", bc_s.sS);

  printf("  On B ");
  if(bc_s.sB == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sBD);
  else if(bc_s.sB == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sBN);
  else if(bc_s.sB == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
  else printf(" bc_s.sB is wrong with value %d\n", bc_s.sB);

  printf("  On T ");
  if(bc_s.sT == DIRICHLET) printf("DIRICHELT BOUNDARY CONDITION %f\n", bc_s.sTD);
  else if(bc_s.sT == NEUMANN) printf("NEUMANN BOUNDARY CONDITION %f\n", bc_s.sTN);
  else if(bc_s.sT == PERIODIC) printf("PERIODIC BOUNDARY CONDITION\n");
  else printf(" bc_s.sT is wrong with value %d\n", bc_s.sT); 
}

void scalar_init(void)
{
  if(scalar_on == 1){
    // allocate and initialize scalar field
    s0 = (real*) malloc(Dom.Gcc.s3b * sizeof(real));
    cpumem += Dom.Gcc.s3b * sizeof(real);
    s = (real*) malloc(Dom.Gcc.s3b * sizeof(real));
    cpumem += Dom.Gcc.s3b * sizeof(real);

    conv0_s = (real*) malloc(Dom.Gcc.s3b * sizeof(real));
    cpumem += Dom.Gcc.s3b * sizeof(real);
    conv_s = (real*) malloc(Dom.Gcc.s3b * sizeof(real));
    cpumem += Dom.Gcc.s3b * sizeof(real);

    diff0_s = (real*) malloc(Dom.Gcc.s3b * sizeof(real));
    cpumem += Dom.Gcc.s3b * sizeof(real);
    diff_s = (real*) malloc(Dom.Gcc.s3b * sizeof(real));
    cpumem += Dom.Gcc.s3b * sizeof(real);

    // initialize QUIESCENT scalar field (default)
    
    for(int i = 0; i < Dom.Gcc.s3b; i++) {
      s0[i] = 0.0;
      s[i] = 0.0;
      conv0_s[i] = 0.0;
      conv_s[i] = 0.0;
      diff0_s[i] = 0.0;
      diff_s[i] = 0.0;
    }
    
    // initialize some variables
    real dx_min = Dom.dx;
    if(Dom.dy < dx_min) dx_min = Dom.dy;
    if(Dom.dz < dx_min) dx_min = Dom.dz;
    dt = 2. * s_d / (dx_min * dx_min);
    dt += 2. * s_d / (Dom.dy * Dom.dy);
    dt += 2. * s_d / (Dom.dz * Dom.dz);
    dt = CFL / dt;
    dt0 = -1.;
  }    
}  

void scalar_clean(void)
{
  if(scalar_on == 1) {
    free(s0);
    free(s);
    free(conv0_s);
    free(conv_s);
    free(diff0_s);
    free(diff_s);
  }
}

void scalar_out_restart(void)
{
  if(scalar_on == 1) {
  // create the file
  char path[FILE_NAME_SIZE] = "";
  sprintf(path, "%s/input/restart_scalar.config", ROOT_DIR);
  FILE *rest = fopen(path, "w");
  if(rest == NULL) {
    fprintf(stderr, "Could not open file restart.input.\n");
    exit(EXIT_FAILURE);
  }
 
  fwrite(s, sizeof(real), Dom.Gfx.s3b, rest);     
  fwrite(s0, sizeof(real), Dom.Gfx.s3b, rest);

  // close the file
  fclose(rest);
  }
}

void scalar_in_restart(void)
{
  if(scalar_on == 1) {
  int fret = 0;
  fret = fret; // prevent compiler warning
  // open configuration file for reading
  char fname[FILE_NAME_SIZE] = "";
  sprintf(fname, "%s/input/restart_scalar.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }  
  
  fret = fread(s, sizeof(real), Dom.Gfx.s3b, infile);
  fret = fread(s0, sizeof(real), Dom.Gfx.s3b, infile);

  // close file
  fclose(infile);
  }
}
























 

