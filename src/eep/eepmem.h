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

#ifndef EEPMEM_H
#define EEPMEM_H
#define RCS_EEPMEM_H "$RCSFile: eepmem.h,v $ $Revision: 2415 $"

#include <stdlib.h>

void *v_calloc(size_t nmemb, size_t size, const char *mtypefrag);
void *v_malloc(size_t size, const char *mtypefrag);
void *v_realloc(void *ptr, size_t size, const char *mtypefrag);

#define v_new(type) (type *) v_malloc(sizeof(type), "v_new")

#define v_free(ptr)   if((ptr) != NULL) { free(ptr); (ptr) = NULL; } 


#define v_extend(ptr, num, type, extnum) \
  (type *) v_realloc((ptr), ((size_t) (num) + (size_t) (extnum)) * sizeof(type), "ext")

/*
  allocate strlen(s) + extlen + 1 bytes and strcpy s into buffer
  return ptr to buffer
*/
char *v_strnew(const char *s, int extlen);
char *v_strcat(char *s1, const char *s2, int extlen);


/**
 * Function to copy and reorder memory.
 * Use this function for an array which you want in another order.
 * 
 * when wanting to move an array (with ints) of size 4 to size 3: [0,1,2,3] -> [3,2,1]
 * use reorder_memory(target, source, sizeof(int), [3,2,1], 3)
 *
 * When using reorder_memory with reorderv[2] = 10
 *   then target[2] = source[10].
 *
 * When using reorder_memory_back with reorderv[2] = 10
 *   then target[10] = source[2];
 */
void reorder_memory(void* target, void* source, size_t memsize, int* reorderv, int reorderc);
void reorder_memory_back(void* target, void* source, size_t memsize, int* reorderv, int reorderc);
  


/*
  floating point matrices (initialized to zero)
*/
float **v_malloc_s2d(int i1, int i2);
void  v_free_s2d(float **m, int i1);

float ***v_malloc_s3d(int i1, int i2, int i3);
void  v_free_s3d(float ***m, int i1, int i2);

double  **v_malloc_d2d(int i1, int i2);
void    v_free_d2d(double **m, int i1);

double  ***v_malloc_d3d(int i1, int i2, int i3);
void    v_free_d3d(double ***m, int i1, int i2);


#endif
