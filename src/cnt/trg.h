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

#ifndef TRG_H
#define TRG_H
#define RCS_TRG_H "$RCSfile: trg.h,v $ $Revision: 2415 $"

#include <stdio.h>
#include <eep/stdint.h>

#include <eep/eepmisc.h>
#include <eep/winsafe.h>

/*
  EEP 2.0 flag word definitions
  used to translate the flags into the EEP 3.x table
*/
#define EEP20_FILTER   0x0200 /* just mentioned, unused in EEP 3.x */
#define EEP20_DISCONT  0x0400
#define EEP20_DCRESET  0x0800
#define EEP20_TRGVAL   0x00ff
#define EEP20_TRGMASK  0x0cff /* to detect all flags used as triggers */

/* 
  a EEP 3.x trigger is a sample number connected with a 8 character code
  codes are stored case-sensitive, but evaluated case-insensitive
*/

#define TRG_CODE_LENGTH 8

/* resets and discontinuity marks are predefinend and have a special handling */
#define TRG_DCRESET "Rs"
#define TRG_DISCONT "__"

#define TRG_IS_DCRESET(code) (!strcasecmp((code), TRG_DCRESET))
#define TRG_IS_DISCONT(code) (!strcasecmp((code), TRG_DISCONT))

typedef char trgcode_t [TRG_CODE_LENGTH + 2]; /* waste one byte to avoid odd length */

typedef struct {
  uint64_t   sample;
  trgcode_t  code;
  char cls_code;
} trgentry_t;

typedef struct {
  asciiline_t   extra_header_text;
  uint64_t      c;
  trgentry_t  * v;
  uint64_t      cmax;
} trg_t;

trg_t *trg_init(void);
void   trg_free(trg_t *trg);

trg_t* trg_copy(const trg_t* trg);

/*
  to initialize the header (cls_range ...), use the following function after trg_init()
*/
void trg_init_header(trg_t *trg, asciiline_t line);

/*
  read and build or write trigger memory table
  caller is responsible for valid streams and cnt data
  return : read: pointer to table or NULL if failed
           write: 0 if successful
*/
trg_t *trg_file_read  (FILE *f, double period);
int    trg_file_write (trg_t *trg, FILE *f, double period, short chanc);

trg_t *trg_file_read_unchecked(FILE *f, double *period, short *chanc);

/*
  insert / delete trigger in table
  set avoids identical triggers but allows multiple triggers in one sample
  code is evaluated case insensitive and stored case sensitive
  
  return: number of cleared/set triggers
*/
int trg_set  (trg_t *trg, uint64_t sample, const char *code);
int trg_clear(trg_t *trg, uint64_t sample, const char *code);

/*
  trg_set as above, but also set cls char
*/
int trg_set_cls(trg_t *trg, uint64_t sample, const char *code, const char cls);

/*
  converts the EEP20 flag word (16 bit) to trigger table entrys
  caller should test the flag contents (EEP20_TRGMASK) and avoid 
  calls without effect
  return: number of added trigger entrys
*/
int trg_set_EEP20(trg_t *trg, uint64_t sample, unsigned short flag);

/*
  full trigger loops
  get the total count and go on step by step
*/
int   trg_get_c(const trg_t *trg);
char *trg_get  (const trg_t *trg, int i, uint64_t *sample);
/*
  trg_get as above, but also set cls char
*/
char *trg_get_cls  (trg_t *trg, int i, uint64_t *sample, char *cls);

/*
  check whether one in a list of triggers matches a trigger
  return: 1 if found
*/
int trg_group_match(char *code, trgcode_t *grpv, short grpc);
int trg_discont_epoch(trg_t *trg, uint64_t start, uint64_t length);

/* 
  look for next(direction 1) or previous(direction 0) trigger <code> in table
  return: trigger index or -1 if not found 
*/
int  trg_seek(trg_t *trg, uint64_t sample, 
                  const char *code, char direction);
int  trg_group_seek(trg_t *trg, uint64_t sample, 
                       trgcode_t *grpv, int grpc, char direction);


#endif
