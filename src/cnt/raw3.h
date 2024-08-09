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

#ifndef RAW3_H
#define RAW3_H
#define RCS_RAW3_H "$RCSfile: raw3.h,v $ $Revision: 2415 $"

#include <eep/eepmisc.h>

/* compression parameters */
#define RAW3_METHODC 3
typedef struct {
  int  method;
  int  nbits;
  int  nexcbits;
  int  length;
  int  hst[33];
  sraw_t *res;
} raw3res_t;

typedef struct {
  short chanc;       /* channel sequence */
  short *chanv;
  
  raw3res_t rc[RAW3_METHODC];    /* some working buffers */
  sraw_t    *last;
  sraw_t    *cur;
} raw3_t;

/* set Verbose on for raw3 error checking (use with care!) */
void raw3_setVerbose(int onoff);
/* for each epoch, the ERR_FLAG_EPOCH can be set */
short raw3_get_ERR_FLAG_EPOCH();
/* therefore, reset using the following function */
void  raw3_set_ERR_FLAG_EPOCH(short);

/*
  prepare data compression for chanc * length signal data blocks
  (the chanv vector is copied)
*/

raw3_t *raw3_init(int chanc, short *chanv, uint64_t length);
void    raw3_free(raw3_t *raw3);

/*
  compress a raw sample matrix to an output buffer
  using the specified channel sequence to allow prediction by neighbor
  caller must supply enough output space - use RAW3_EPOCH_SIZE
  return: number of bytes filled in output buffer;

  input buffer is in pure MUX format here (no EEP 2.0 control words!):
  memory address->
  |chan0 sample0| chan1 sample0|chan2 sample0...
  |chan0 sample1| chan1 sample1|chan2 sample1...
  ...

  (maybe I add all functions for time series buffers later)
*/

#define RAW3_EPOCH_SIZE(length, chanc) ((length + 2) * (chanc) * 4)

int compepoch_mux(raw3_t *raw3, sraw_t *in, int length, char *out);

/*
  the same job backwards
  return: number of bytes used from input buffer
*/ 

int decompepoch_mux(raw3_t *raw3, char *in, int length, sraw_t *out);

/*
  find and return a good channel sequence for prediction by neighbor 
  buf contains the multiplexed raw data to predict from and is unchanged
  chanv space (chanc elements) has to be supplied by caller
*/
void compchanv_mux(sraw_t *buf, int length, 
                    short chanc, short *chanv);

#endif
