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

#ifndef REJ_H
#define REJ_H
#define RCS_REJ_H "$RCSfile: rej.h,v $ $Revision: 2415 $"

#include <stdio.h>
#include <eep/stdint.h>
#include <eep/eepmisc.h>

typedef struct {
  uint64_t start;           /* in samples */
  uint64_t length;
} rejentry_t;

typedef struct {
  int        c;           /* rejection epoch count */
  rejentry_t *v;          /* rejection epoch vector */
} rej_t;

rej_t *rej_init(void);
void   rej_free(rej_t *rej);

/* 
  init rejection table by fetching .rej file
  return: table or NULL on error
*/
rej_t *rej_file_read(FILE *f, double period);

/*
  dump rejection table contents to ascii file
  return: 0 on success
*/
int   rej_file_write(rej_t *rej, FILE *f, double period);

/*
  OR rejection epoch into table
*/
void  rej_set(rej_t *rej, uint64_t start, uint64_t length);
void  rej_clear(rej_t *rej, uint64_t start, uint64_t length);

/*
  retrieve rejections
*/
int   rej_get_c(rej_t *rej);
void  rej_get(rej_t *rej, int i, uint64_t *start, uint64_t *length);

int   rej_seek(rej_t *rej, uint64_t start, char direction);

/*
  ask for present rejections
*/
int is_rejected(rej_t *rej, uint64_t sample);
int is_rejected_epoch(rej_t *rej, uint64_t sample, uint64_t length);

#endif
