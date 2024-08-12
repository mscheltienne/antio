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
 
#ifndef EEPMISC_H
#define EEPMISC_H
#define RCS_EEPMISC_H "$RCSfile: eepmisc.h,v $ $Revision: 2415 $"

#include <stdio.h>
#include <eep/stdint.h>

typedef int  sraw_t;   /* "sample raw" - a type which can contain 32 bit signeds */
typedef int  slen_t;   /* "sample length" - signed, must count sample numbers (up to 1e7) */

typedef struct {      /* "sample window" */
  int64_t start;
  int64_t length;
} swin_t;


#define ABS(x) (((x) < 0) ? -(x) : (x))
#define FRND(x) ((x) < 0 ? ((x) - 0.5) : ((x) + 0.5))

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#if defined(__EMX__) || (defined(sun) && defined(c_plusplus))
int strcasecmp(const char *s1, const char *s2);
#endif

/* returns 1 if s ends with m */
int strend(char *s, char *m);

/* emulate some ANSI C definitions (made for a Sparc/SunOS 4.1.4) */
#ifndef SEEK_SET
# define SEEK_SET 0
#endif
#ifndef SEEK_CUR
# define SEEK_CUR 1
#endif
#ifndef SEEK_END
# define SEEK_END 2
#endif

/* ------------------------------------------------------------------
  text file handling
*/
#define EEPIOMAX 1024
typedef char asciiline_t[EEPIOMAX];

/*
  you must supply at least EEPIOMAX chars for reading in line
  an asciiline_t buffer will suffice
  return: fgets return (NULL on EOF or error, else line)
*/
char *asciiread(FILE *f, char *line);

/*
  return: nonzero on success
*/
int  asciiwrite(FILE *f, char *line);

#endif
