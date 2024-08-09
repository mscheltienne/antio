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

/* source not (yet) available for native WIN32 */
#if ! defined(__CYGWIN__)

#ifndef AVRCFG_H
#define AVRCFG_H
#define RCS_AVRCFG_H "$RCSfile: avrcfg.h,v $ $Revision: 2415 $"

#include <cnt/cnt.h>

#define AVR_IGNORE_NOTHING   0x0000
#define AVR_IGNORE_REJECTION 0x0001
#define AVR_IGNORE_AVRWINDOW 0x0010


#define EEPCOLORC 33
extern char eepcolortbl[EEPCOLORC + 1][11];
int cfg_get_eepcolor(char colorstr[11]);

#define AVR_COLOR_NR_TO_INDEX(x) ((x) > 15 ? (x - 7) : (x - 8))

char *cfg_put_eepcolorstr(int color, char *colorstr);
char *get_x_colorstring(const char *avrcolor);

/* average -------------------------------------------------------- */

typedef struct {
  short     codec;
  trgcode_t *codetbl;  /* all triggers for this condition */
  trgcode_t  code;     /* new condition shortcut */
  char lab[16];
  char col[10];
} avrcondition_t;

typedef struct {
  short           condc;
  avrcondition_t  *condtbl;
  short  chanc;
  short  *chantbl;
  
  int       iswindow;
  swin_t    window;
  
  int       isbaseline;
  swin_t    baseline;
  short     refdisp;
  trgcode_t     *reftrgv;
  
  int       isrejection;  
  swin_t    rejection;
  int       isbslrejection;
  swin_t    bslrejection;  /* added for use in case of reference displacement 
                              - MG  07/03/2000  */
} AverageParameters;

int ReadAverageParameters(FILE *Cfg, eeg_t *EEG_p, AverageParameters *p);
void FreeAverageParameters(AverageParameters *p);

/* mode: AVR_IGNORE_(NOTHING|REJECTION|AVRWINDOW) */
void ShowAverageParameters(AverageParameters p, eeg_t *src, short mode);   
int check_reject_window_settings(AverageParameters p);

/* retrieve information strings from the AverageParameters struct 
   (strGet.. functions) */ 
/* get whitespace-separated list of condition triggers */
char *strGetConditionTriggers( AverageParameters *cfg, short condI );

/* get reference trigger, 
   return NULL if no baseline used or undisplaced reference */
char *strGetReferenceTrigger( AverageParameters *cfg, short condI );
char *strGetAllReferenceTriggers( AverageParameters *cfg );

/* return window settings in ms (e.g. "-200..1500"),
   return NULL if window not used */
char *strGetAverageWindow( eeg_t *eeg, AverageParameters *cfg );
char *strGetBaselineWindow( eeg_t *eeg, AverageParameters *cfg );
char *strGetRejectionWindow( eeg_t *eeg, AverageParameters *cfg );
char *strGetBslRejectionWindow( eeg_t *eeg, AverageParameters *cfg );


#endif

#endif /* ! defined(WIN32) || defined(__CYGWIN__) */
