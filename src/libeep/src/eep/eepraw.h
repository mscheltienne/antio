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

#ifndef EEPRAW_H
#define EEPRAW_H
#define RCS_EEPRAW_H "$RCSfile: eepraw.h,v $ $Revision: 2415 $"

#include <eep/stdint.h>
#include <stdio.h>

#include <eep/eepmisc.h>

/* supported architecture byteorders */
#define EEP_LITTLE_ENDIAN 1234
#define EEP_BIG_ENDIAN    4321

/*
  the following preprocessor magic ends up in symbols
    EEP_BYTE_ORDER
    EEP_FLOAT_ORDER
  which simply indicate whether integer and/or float types
  must be byteswapped on load or not
  eep file data are always little endian - greetings from 80x86/MS-DOS :-)
*/

#ifndef EEP_BYTE_ORDER

#if defined(sparc) || defined(__sparc) || defined(__mips) || defined(__POWERPC__) || defined(__powerpc__)
#define EEP_BYTE_ORDER  EEP_BIG_ENDIAN
// #warning "using: EEP_BIG_ENDIAN"
#else  /* if defined(__alpha) || defined(__i386) || defined(i386) */
#define EEP_BYTE_ORDER  EEP_LITTLE_ENDIAN
// #warning "using: EEP_LITTLE_ENDIAN"
#endif

#endif /* EEP_BYTE_ORDER */


#ifndef EEP_FLOAT_ORDER

#if defined(sparc) || defined(__sparc) || defined(__mips) || defined(__POWERPC__) || defined(__powerpc__)
#define EEP_FLOAT_ORDER EEP_BIG_ENDIAN
#else  /* if defined(__alpha) || defined(__i386) || defined(i386) */
#define EEP_FLOAT_ORDER EEP_LITTLE_ENDIAN
#endif

#endif /* EEP_FLOAT_ORDER */


/*
  to read/write one item from/to file, (in several binary headers)
  return: 0 on error, 1 on success (fread/fwrite return)
*/
int read_s64 (FILE *f, int64_t *v);
int read_u64 (FILE *f, uint64_t *v);
int read_s32 (FILE *f, int *v);
int read_u32 (FILE *f, unsigned int *v);
int read_s16 (FILE *f, int *v);
int read_u16 (FILE *f, int *v);
int read_f32 (FILE *f, float *v);
int read_f64 (FILE *f, double *v);

int write_s64  (FILE *f, int64_t v);
int write_u64  (FILE *f, uint64_t v);
int write_s32  (FILE *f, int v);
int write_u32  (FILE *f, unsigned int v);
int write_s16  (FILE *f, int v);
int write_u16  (FILE *f, int v);
int write_f32  (FILE *f, float v);
int write_f64(FILE *f, double v);

/*
  to write one item into memory
*/
void swrite_u64  (char *s, uint64_t v);
void swrite_s32  (char *s, int v);
void swrite_s16  (char *s, int v);

void swrite_f32  (char *s, float v);
void sread_f32   (char *s, float *v);

void swrite_f64   (char *s, double v);

/*
  read/write vectors (for NS/EEP 2.0 cnt files and avr files)
  write may change the buffer contents !

  return: number of read/writted items
*/
int vread_s16 (FILE *f, sraw_t *buf, int n);
int vwrite_s16(FILE *f, sraw_t *buf, int n);

int vread_f32 (FILE *f, float *buf, int n);
int vwrite_f32(FILE *f, float *buf, int n);

int vread_s32(FILE *f, sraw_t *buf, int n);

#endif
