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

#include <string.h>
#include <ctype.h>

#include <cnt/rej.h>
#include <eep/eepio.h>
#include <eep/eepmem.h>

#ifdef COMPILE_RCS
char RCS_rej_h[] = RCS_REJ_H;
char RCS_rej_c[] = "$RCSfile: rej.c,v $ $Revision: 2415 $";
#endif

#define REJ_OK  0
#define REJ_EOF 1
#define REJ_ERR 2

int rej_line_write(FILE *Rej, double period, uint64_t first, uint64_t last)
{
   asciiline_t line;
   asciiline_t col2;
   
   sprintf(line, "%.6f-", first * period);
   sprintf(col2, "%.6f\n", last * period);
   strcat(line, col2);
   asciiwrite(Rej, line);

   return 0;
}

int rej_line_parse(char *rejline, double period, uint64_t *first, uint64_t *last)
{
  double t1, t2;
  int i;
  char *c;
  
  i = sscanf(rejline, "%lf", &t1);
  if (!i)
    return 1;
  c = (char *) strrchr(rejline, '-');
  if (!c)
    return 2;
  c++;
  i = sscanf(c, "%lf", &t2);
  if (!i)
    return 3;
  
  *first = (uint64_t) (t1 / period + 0.5);
  *last = (uint64_t) (t2 / period + 0.5);
  if (*first > *last || *last < 0)
    return 4;
    
  return 0;
}

char *rej_line_norm(char *line)
{
  char *tmp;
  char *buf;
  int i = 0;
  int j = 0;
  int c;


  buf = v_strnew(line, 0);
  if ((tmp = (char *) strchr(buf, ';')))
    *tmp='\0';                                    /* cut comment if exists */
  while ((c = buf[i++]))
    if (!isspace(c)) 
      line[j++] = toupper(c);                     /* copy valid characters */
  line[j]='\0';
  free(buf);

  return line;
}

int rej_line_read(FILE *Rej, double period, uint64_t *first, uint64_t *last)
{
  char *c;
  asciiline_t rejline;
  int r = REJ_OK;
  
  do {
    c = asciiread(Rej, rejline);
    rej_line_norm(rejline);
  } while (!feof(Rej) && c != NULL && rejline[0] == '\0');
  
  if (!feof(Rej)) {
    if (c == NULL)
      r = REJ_ERR;
    if (rej_line_parse(rejline, period, first, last))
      r = REJ_ERR;
  }

  else {
    r = REJ_EOF;
  }

  return r;
}

/* public ----------------------------------------- */

rej_t *rej_init(void)
{
  rej_t *rej = (rej_t *) v_malloc(sizeof(rej_t), "rej");
  rej->v = NULL;
  rej->c = 0;
  return rej;
}

void rej_free(rej_t *rej)
{
  if (rej) {
    v_free(rej->v);
    v_free(rej);
  }
}

rej_t *rej_file_read(FILE *f, double period)
{
  rej_t  *rej = rej_init();
  uint64_t first, last;
  int    status;
  
  rewind(f);
  
  while (!feof(f)) {
    status = rej_line_read(f, period, &first, &last);
    if (status == REJ_ERR) {
      rej_free(rej);
      return NULL;
    }
    else if (status == REJ_OK) {
      if (rej->c % 64 == 0) {
        rej->v = v_extend(rej->v, rej->c, rejentry_t, 64);
      }

      rej->v[rej->c].start = first;
      rej->v[rej->c].length = last + 1 - first;
      (rej->c)++;
    }
  }
  return rej;
}

int rej_file_write(rej_t *rej, FILE *f, double period)
{
  int i;
  
  rewind(f);
  for (i = 0; i < rej->c; i++) {
    if (rej_line_write(f, period, 
        rej->v[i].start, rej->v[i].start + rej->v[i].length - 1))
    {
      return 1;
    }
  }
  return 0;
}

void rej_set(rej_t *rej, uint64_t start, uint64_t length)
{
  int *rejc =   &(rej->c);
  rejentry_t **rejv = &(rej->v);

  int i = 0;
  int overlaps = 0;

  uint64_t effstart, efflength;

  /* ignore all rejections completely before new start */
  while (i < *rejc && (*rejv)[i].start + (*rejv)[i].length < start) i++;

  /* count all rejections touched by the new one */

  if ((i > 0) && (start <= (*rejv)[i-1].start + (*rejv)[i-1].length)) {
    overlaps++;
  }

  while ( i + overlaps < *rejc 
         && start + length >= (*rejv)[i + overlaps].start)
  {
    overlaps++;
  }

  if (overlaps) {
    effstart = MIN((*rejv)[i].start, 
                  start);
    efflength = MAX((*rejv)[i + overlaps - 1].start + (*rejv)[i + overlaps - 1].length, 
                      start + length)
                - effstart;
  }
  else {
    effstart = start;
    efflength = length;
  }
    
  /* nothing touched - insert a new rejection in table*/
  if (overlaps == 0) {
    *rejv = (rejentry_t *)
      v_realloc(*rejv, (size_t) (*rejc + 1) * sizeof(rejentry_t), "rejv");
    memmove(&(*rejv)[i + 1], &(*rejv)[i], 
            (size_t) (*rejc - i) * sizeof(rejentry_t));
    (*rejc)++;
  }
  
  /* exact one touched - nothing needed in table, change size only */
  
  /* many touched - merge them to one */
  if (overlaps > 1) {
    if (*rejc - i - overlaps + 1 > 0) {
      memmove(&(*rejv)[i], &(*rejv)[i + overlaps - 1], 
              (size_t) (*rejc - i - overlaps + 1) * sizeof(rejentry_t));
    }
    *rejv = (rejentry_t *)
      v_realloc(*rejv, (size_t) (*rejc - overlaps + 1) * sizeof(rejentry_t), "rejv");
    (*rejc) -= overlaps - 1;
  }

  (*rejv)[i].start = effstart;
  (*rejv)[i].length = efflength;
}

int rej_get_c(rej_t *rej)
{
  return rej->c;
}

void rej_get(rej_t *rej, int i, uint64_t *start, uint64_t *length)
{
  *start = rej->v[i].start;
  *length = rej->v[i].length;
}

int is_rejected(rej_t *rej, uint64_t sample)
{
  int r = 0;
  static int i = 0;
  int rejc =   rej->c;
  rejentry_t *rejv = rej->v;
  
  /* have to rewind the static counter ? */
  if (i >= rejc) 
    i=0;   /* added M.G.*/
  
  while (i > 0 && i < rejc && rejv[i].start > sample)
    i--;
  
  /* locate first rejection which ends after interesting sample */
  while (i < rejc && rejv[i].start + rejv[i].length - 1 < sample)
    i++;
  
  /* found any ? - check whether it covers the sample */
  if (i < rejc && rejv[i].start <= sample)
    r = 1;
  
  return r;
}

int is_rejected_epoch(rej_t *rej, uint64_t sample, uint64_t length)
{
  int r = 0;
  int i = 0;
  int rejc =   rej->c;
  rejentry_t *rejv = rej->v;
  
  /* locate first rejection which ends after first interesting sample */
  while (i < rejc && rejv[i].start + rejv[i].length - 1 < sample) 
    i++;
  
  /* found any ? - check whether it touches the epoch */
  if (i < rejc && rejv[i].start < sample + length)
    r = 1;
  
  return r;
}


int rej_seek(rej_t *rej, uint64_t start, char direction)
{
  int rejc = rej->c;
  rejentry_t *rejv = rej->v;
  
  int i = 0;
  int lasti = -1;
  
  while (i < rejc && rejv[i].start <= start) {
    if (rejv[i].start < start) lasti = i;
    i++;
  };
  
  if (direction)
    return i < rejc ? i : -1;
  else
    return lasti;
}


void rej_clear(rej_t *rej, uint64_t start, uint64_t length)
{
  int *rejc =   &(rej->c);
  rejentry_t **rejv = &(rej->v);

  int i = 0;
  int del = 0;
  uint64_t tmp;
  
  /* ignore until deletion starts */
  while (i < *rejc && (*rejv)[i].start + (*rejv)[i].length < start) i++;
  
  if (i < *rejc) {
  
    /* split rejection ? */
    if (   (*rejv)[i].start < start 
        && (*rejv)[i].start + (*rejv)[i].length > start + length)
    {
      *rejv = (rejentry_t *)
        v_realloc(*rejv, (size_t) (*rejc + 1) * sizeof(rejentry_t), "rejv");
      memmove(&(*rejv)[i + 1], &(*rejv)[i], (size_t) (*rejc - i) * sizeof(rejentry_t));
      (*rejv)[i + 1].start = start + length;
      (*rejv)[i + 1].length =   (*rejv)[i].start + (*rejv)[i].length 
                                - (start + length);
      (*rejv)[i].length = start - (*rejv)[i].start;
      (*rejc)++;
    }
    
    else {

      /* delete latter part of rejection ? */
      if (i < *rejc) {
        if ((*rejv)[i].start < start) {
          (*rejv)[i].length = start - (*rejv)[i].start;
          i++; /* this one is handled - locate next in table */
        }
      }

      /* delete complete rejections ? */
      while (   ((i + del) < *rejc)
             && ((*rejv)[i + del].start 
                >= start)
             && ((*rejv)[i + del].start + (*rejv)[i + del].length
                <= start + length))
      {
        del++;
      }
      if (del) {
        memmove(&(*rejv)[i], &(*rejv)[i + del],
          (size_t) (*rejc - i - del) * sizeof(rejentry_t));
        *rejv = (rejentry_t *)
          v_realloc(*rejv, (size_t) (*rejc - del) * sizeof(rejentry_t), "rejv");
        (*rejc) -= del;
      }
      
      /* delete front part of next rejection left ? */
      if (i < *rejc) {
        if ((*rejv)[i].start < start + length) {
          tmp = (*rejv)[i].start;
          (*rejv)[i].start = start + length;
          (*rejv)[i].length += tmp - (start + length);
        }
      }
    }
  }
}

