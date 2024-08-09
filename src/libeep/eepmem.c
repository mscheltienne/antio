/********************************************************************************
 *                                                                              *
 * this file is part of:                                                        *
 * libeep, the project for reading and writing avr/cnt eeg and related files    *
 *                                                                              *
 ********************************************************************************
 *                                                                              *
 * LICENSE:Copyright (c) 2003-2009,                                             *
 * Advanced Neuro Technology (ANT) B.V., Enschede, The Netherlands              *
 * Max-Planck Institute for Human Cognitive & Brain Sciences, Leipzig, Germany  *
 *                                                                              *
 ********************************************************************************
 *                                                                              *
 * This library is free software; you can redistribute it and/or modify         *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation; either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This library is distributed WITHOUT ANY WARRANTY; even the implied warranty  *
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              *
 * GNU Lesser General Public License for more details.                          *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program. If not, see <http://www.gnu.org/licenses/>          *
 *                                                                              *
 *******************************************************************************/

#include <stdio.h>
#include <string.h>

#include <eep/eepmem.h>
#include <eep/eepio.h>

#ifdef COMPILE_RCS
char RCS_eepmem_h[] = RCS_EEPMEM_H;
char RCS_eepmem_c[] = "$RCSFile: eepmem.c,v $ $Revision: 2415 $";
#endif

#ifdef BYPASS_V_FUNCTIONS
void *v_calloc(size_t nmemb, size_t size, const char *mtypefrag) {
  // fprintf(stderr, "%s, %s\n", __FUNCTION__, mtypefrag);
  return calloc(nmemb, size);
}
void *v_malloc(size_t size, const char *mtypefrag) {
  // fprintf(stderr, "%s, %s\n", __FUNCTION__, mtypefrag);
  return malloc(size);
}
void *v_realloc(void *ptr, size_t size, const char *mtypefrag) {
  // fprintf(stderr, "%s, %s\n", __FUNCTION__, mtypefrag);
  return realloc(ptr, size);
}
#else
void *v_calloc(size_t nmemb, size_t size, const char *mtypefrag)
{
  void *p;
  
  if ( nmemb == 0 ) return NULL;
  p = (void *) calloc(nmemb, size);
  if (p == NULL) 
    eeperror("libeep: failed to callocate %s memory (%ld bytes)!\n", 
            mtypefrag, (unsigned long) size * nmemb);
  return p;
}

void *v_malloc(size_t size, const char *mtypefrag)
{
  void *p;
  
  if(size == 0) return NULL;
  p = (void *) malloc(size);
  if (p == NULL) 
    eeperror("libeep: failed to mallocate %s memory (%ld bytes)!\n", 
            mtypefrag, (unsigned long) size);
  return p;
}

void *v_realloc(void *ptr, size_t size, const char *mtypefrag)
{
  void *p = NULL;
  
  if (ptr == NULL && size != 0) {
    p = (void *) malloc(size);
  }
  if (ptr != NULL) { 
    p = (void *) realloc((char *) ptr, size);
  }
  
  if (p == NULL && size != 0) 
    eeperror("libeep: failed to reallocate %s memory (%ld bytes)!\n", 
            mtypefrag, (unsigned long) size);

  if (size == 0) p = NULL;
  return p;
}

#endif

char *v_strnew(const char *s, int extlen)
{
  char *p;
  
  p = (char *) malloc(strlen(s) + extlen + 1);
  if (p == NULL) 
    eeperror("libeep: failed to strnew %ld bytes!\n", 
            (unsigned long) (strlen(s) + extlen + 1));
  strcpy(p, s);
  return p;
}

char *v_strcat(char *s1, const char *s2, int extlen)
{
  s1 = (char *) realloc(s1, strlen(s1) + strlen(s2) + extlen + 1);
  if (s1 == NULL) 
    eeperror("libeep: failed to strcat %ld bytes!\n", 
            (unsigned long) (strlen(s1) + strlen(s2) + extlen + 1));
  strcat(s1, s2);
  return s1;
}


void reorder_memory(void* target, void* source, size_t memsize, int* reorderv, int reorderc)
{
  int i;
  for(i=0;i<reorderc; i++)
  {
    if(reorderv[i] != -1)
      memcpy((void*)((char*)target + (memsize*i)), (void*)((char*)source + (memsize * reorderv[i])), memsize);
  }
}

void reorder_memory_back(void* target, void* source, size_t memsize, int* reorderv, int reorderc)
{
  int i;
  for(i=0;i<reorderc; i++)
  {
    if(reorderv[i] != -1)
      memcpy((void*)((char*)target + (memsize* reorderv[i])), (void*)((char*)source + (memsize * i)), memsize);
  }
}


float **v_malloc_s2d(int i1, int i2)
{
  int i, j;

  float **v = (float **) v_malloc(i1 * sizeof(float *), "s2d");
  for (i = 0; i < i1; i++) {
    v[i] = (float *) v_malloc(i2 * sizeof(float), "s2d");
    for (j = 0; j < i2; j++)
      v[i][j] = 0.0;
  }

  return v;
}

void v_free_s2d(float **v, int i1)
{
  int i;
  
  if( v == NULL ) return; 
  
  for (i = i1 - 1; i >= 0; i--)
    v_free(v[i]);
  v_free(v);
} 

float ***v_malloc_s3d(int i1, int i2, int i3)
{
  int i;
  
  float ***v = (float ***) v_malloc(i1 * sizeof(float **), "s3d");
  
  for (i = 0; i < i1; i++)
    v[i] = v_malloc_s2d(i2, i3);

  return v;
}

void v_free_s3d(float ***v, int i1, int i2)
{
  int i;
  
  if( v == NULL ) return;
  
  for (i = i1 - 1; i >= 0; i--)
    v_free_s2d(v[i], i2);
  v_free(v);
}


double **v_malloc_d2d(int d1, int d2)
{
  int    i, j;
  double **m;

  m = (double **) v_malloc(d1 * sizeof(double *), "d2d");
  for (i = 0; i < d1; i++) {
    m[i] = (double *) v_malloc(d2 * sizeof(double ), "d2d");
    for (j = 0; j < d2; j++)
      m[i][j] = 0.0;
  }
  
  return m;
}

void v_free_d2d(double **m, int d1)
{
  int    i;
  
  if( m == NULL ) return;

  for (i = 0; i < d1; i++) {
    v_free(m[i]);
  }
  v_free(m);
}

double ***v_malloc_d3d(int d1, int d2, int d3)
{
  int    i;
  double ***m = (double ***) v_malloc(d1 * sizeof(double **), "d3d");
  for (i = 0; i < d1; i++)
    m[i] = v_malloc_d2d(d2, d3);
  
  return m;
}

void v_free_d3d(double ***m, int d1, int d2)
{
  int    i;

  if( m == NULL ) return;
  
  for (i = 0; i < d1; i++)
    v_free_d2d(m[i], d2);
  v_free(m);
}
