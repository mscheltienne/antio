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

#ifndef AVR_H
#define AVR_H
#define RCS_AVR_H "$RCSfile: avr.h,v $ $Revision: 2415 $"

#include <stdio.h>
#include <eep/stdint.h>
#include <eep/eepmisc.h>

#define AVR_HEADER_SIZE         38
#define AVR_CHANNEL_HEADER_SIZE 16

typedef char chanlab_t[11];

typedef struct {
  char          lab[11];                 /* channel label */
  uint64_t      filepos;                 /* offset of data in file */
} avrchan_t;

typedef struct {
  char           condlab[11];            /* condition label */
  char           condcol[9];             /* associated color code */
                                         /* e.g. "color:23"       */

  unsigned short trialc;                 /* total number of trials */
  unsigned short rejtrialc;              /* number of rejected trials */
  int64_t        sample0;                /* index (with respect to trigger) of first sample */  
  uint64_t       samplec;                /* number of samples */
  float          period;                 /* sampling intervall in seconds */

  float          mtrialc;                /* mean of trial numbers for grand_av */
                                         /* not stored, initialized to trialc - rejtrialc during load */
  
  unsigned short chanc;                  /* number of channels */
  avrchan_t      *chanv;                 /* channel info table */
  
  unsigned short histc;                  /* number of entries in history */
  char          **histv;                 /* history table */
  size_t         hist_size;              /* history length in bytes */
  
  short          header_size;
  short          channel_header_size;
} avr_t;

#define AVRERR_NONE 0
#define AVRERR_FILE 1
#define AVRERR_DATA 2


/*
  fill the avr structure with file data
*/
int avropen  (avr_t *avr, FILE *f);
int avrclose (avr_t *avr);

/* free dynam. alooc. memory, init history struct with safe values */
void free_avr_history(avr_t *avr);

/* append one entry to the history, (allocate history, if not yet present)  */
int append_avr_history( avr_t *avr, const char *line);

/* copy history members */
void copy_avr_history( avr_t *src, avr_t *dst);

/* print avr history members to stdout, wrap lines after linelen chars */
void show_avr_history( avr_t *avr, int linelen );

/*
  insert format specific contents into structure and
  dump the complete structure contents to file
  append the registry and cmdline strings to the file history
*/
int avrnew   (avr_t *avr, FILE *f, const char *registry, const char *cmdline);

/*
  duplicate avr structure, 
  retain / discard avr-history according to last parameter
*/
void avrcopy (avr_t *src, avr_t *dst, short retain_history);

#define AVRBAND_MEAN 0
#define AVRBAND_VAR  1

int avrseek (avr_t *avr, FILE *f, short chan, short band);
int avrread (FILE *f, float *v, uint64_t c);
int avrwrite(FILE *f, float *v, uint64_t c);

short get_avr_headerSize(avr_t *avr);
short get_avr_channelHeaderSize(avr_t *avr);
size_t get_avr_histSize(avr_t *avr);
size_t get_avr_totalHeaderSize(avr_t *avr);

char *get_avr_chan_lab(avr_t *avr, short indx);
unsigned short get_avr_chanc(avr_t *avr);
unsigned short get_avr_trialc(avr_t *avr);
unsigned short get_avr_rejectc(avr_t *avr);
uint64_t get_avr_samplec(avr_t *avr);
float get_avr_period(avr_t *avr);



short avr_eep_get_chan_index(avr_t *avr, char *lab, short try_first);

/*
  optionally, allocate a matrix
  load avr file contents into it
  avr is expected to contain all required data, no checks here!
  chanc   = 0 requests all channels in the original order
  samplec = 0 requests all sample points
  in every case, v must point to sufficient space or to NULL
  
  return: v (if v is NULL on entry, it points to the newly allocated space)
*/
float **avr_load(avr_t *avr, FILE *f, float **v,
                 chanlab_t *chanv, short chanc, uint64_t sample0, uint64_t samplec,
                 int band);
/*
  save average data matrix
  v contents must match avr channel layout
*/
void avr_save(avr_t *avr, FILE *f, float **v, int band);


/*
  load the specified time slice from the avr according to chanv
  avr must contain all requested channels/samples, no checks here!
  return: 0 on success, 1 on read error
*/

int avr_read_slice(avr_t *avr, FILE *Avr, uint64_t start, uint64_t length, 
                    chanlab_t *chanv, short chanc, float *slice);

/*
  return: 1 if the vector contains valid variance information
          0 otherwise
*/
int avr_var_valid(float *v, int c);


#endif
