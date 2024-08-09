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

#include <ctype.h>
#include <string.h>

#include <avr/avrcfg.h>
#include <eep/eepio.h>
#include <eep/eepmem.h>
#include <eep/winsafe.h>

/*
#define MS2SAMPLE(mstime, period) (((float) mstime) / (period) / 1000.0 + 0.5)
*/
#define FRND(x) ((x) < 0 ? ((x) - 0.5) : ((x) + 0.5))

char RCS_avrcfg_h[] = RCS_AVRCFG_H;
char RCS_avrcfg_c[] = "$RCSfile: avrcfg.c,v $ $Revision: 2415 $";

/* general dos eep color definitions */
char eepcolortbl[EEPCOLORC + 1][11] = { "",
  "BLUE", "GREEN", "CYAN", "RED",  "MAGENTA", "YELLOW", "WHITE", "BLACK",
  "BLUE", "STEEL", "SKY",  "CYAN", "MINT",    "SEA",    "LEAVES","GREEN", 
  "OLIVE","SIENNA","LIGHTGREEN","YELLOW","OCHRE", "APRICOT", "ORANGE", "RED", 
  "CRIMSON","ROSE", "PINK", "MAGENTA", "PURPLE", "LILAC", "AUBERGINE", "PLUM",
  "UV"
};


char *get_x_colorstring(const char *avrcolor) 
{
  /* EEP 2.0 colors patched for bright bg and converted to X names */
  static char eepxcolortbl[EEPCOLORC + 1][20] = { "rgb:0000/0000/0000",
    "rgb:0000/0000/ffff", 
    "rgb:0000/ffff/0000", 
    "rgb:0000/ffff/ffff", 
    "rgb:ffff/0000/0000", 
    "rgb:ffff/0000/ffff", 
    "rgb:da00/a500/2000", 
    "rgb:7fff/7fff/7fff", 
    "rgb:0000/0000/0000",
    "rgb:0000/0000/ffff", 
    "rgb:0000/0000/8b00", 
    "rgb:6c00/a600/cd00", 
    "rgb:0000/ffff/ffff",
    "rgb:0000/ff00/7f00", 
    "rgb:8f00/bc00/8f00", 
    "rgb:8b00/8600/4e00",
    "rgb:0000/ffff/0000", 
    "rgb:bc00/ee00/6800",
    "rgb:a000/5200/2d00",
    "rgb:4300/cd00/8000",
    "rgb:da00/a500/2000", 
    "rgb:ff00/8200/4700", 
    "rgb:cd00/ad00/0000", 
    "rgb:ff00/8c00/0000",
    "rgb:ffff/0000/0000", 
    "rgb:cd00/6600/1d00",
    "rgb:9900/3200/cc00", 
    "rgb:ff00/1400/9300", 
    "rgb:ffff/0000/ffff", 
    "rgb:a000/2000/f000",
    "rgb:8a00/2b00/e200",
    "rgb:6600/cd00/aa00",
    "rgb:2700/4000/8b00",
    "rgb:0000/6800/8b00"
  };

  char *colnr = NULL;
  int col = 0;

  /* few conversions are needed to get the color from avr ...*/
  if( avrcolor ) {
    colnr = strchr(avrcolor, ':'); 
  }
  if( colnr ) {
    colnr++;
    sscanf(colnr, "%d", &col);
  }
  col = (col < 8 || col > 40) ? 8 : col;

  return  eepxcolortbl[AVR_COLOR_NR_TO_INDEX(col)];
}

int cfg_get_eepcolor( char colorstr[11])
{
  int i, ret;
  
  i = 0;
  while ((colorstr[i] = toupper(colorstr[i]))) i++;
  
  ret = -1;
  for (i = 1; i <= EEPCOLORC; i++)
    if (!strcmp(eepcolortbl[i], colorstr)) ret = i + 8;
  
  if (ret == 16) ret = 15;
  if (ret > 16) ret--;
  if (ret > 40) ret = 40;
  
  return ret;
}

char *cfg_put_eepcolorstr(int color, char *colorstr)
{
  if (color > 15)
    strcpy(colorstr, eepcolortbl[color - 7]);
  else
    strcpy(colorstr, eepcolortbl[color - 8]);
  
  return colorstr;
}

/* general item parsers ------------------------------------------ */


/*
  get an 2-dot separated interval given in integer msec values:
  -300 .. 1500
  +300..+1200
  
  samples of interval limits BOTH included !!!
*/

int cfg_get_msecint(const char *line, swin_t *win, eeg_t *src)
{
  
  int t1, t2, i;
  int r = 1;
  float period = eep_get_period(src);
  
  i = sscanf(line, "%d..%d", &t1, &t2);
  if (i == 2 && t2 > t1) {
    win->start =   FRND(t1 / 1000.0 / period);
    win->length =  FRND((t2 - t1) / 1000.0 / period);
    (win->length)++;
    r = 0;
  }

  return r;
}


/* average ------------------------------------------------------- */

typedef int (*avrfunc_t) ();            /* cfg item parser prototypes */

int avrkey(char *line, int mode, AverageParameters *p);
int avrcond(char *line, int mode, AverageParameters *p);
int avrcondlab(char *line, int mode, AverageParameters *p);
int avrchan(char *line, int mode, AverageParameters *p, eeg_t *e);
int avrwin(char *line, int mode, AverageParameters *p, eeg_t *e);
int avrbase(char *line, int mode, AverageParameters *p, eeg_t *e);
int avrrej(char *line, int mode, AverageParameters *p, eeg_t *e);
int avrbslrej(char *line, int mode, AverageParameters *p, eeg_t *e);

enum avrmodes {                           /* order of next three matters! */
  AVRKEY = 0,                             /* cfg item codes */
  AVRCOND,
  AVRCONDLAB,
  AVRCHAN,
  AVRWIN,
  AVRBASE,
  AVRBSLREJ,
  AVRREJ,  
  AVRERROR
};
#define avrmodesc AVRERROR

avrfunc_t avrfunc[] = {                /* a parse function for each mode */
  &avrkey,
  &avrcond,
  &avrcondlab,
  &avrchan,
  &avrwin,
  &avrbase,
  &avrbslrej,
  &avrrej  
};  

char *avrkeyword[] = {                 /* a keyword which enters its mode */
  ";;;",                                  
  "COND",
  ";;;",
  "CHAN",
  "WINDOW",
  "BASELINE",
  "BSLREJ",
  "REJECT"
  
};

/* 
  every parser functions know the cfg syntax
  and the members it have to set up
*/

int avrkey(char *line, int mode, AverageParameters *p)  
{
  int i;

  for (i = 0; i < avrmodesc; i++) {    
    if (strstr(line, avrkeyword[i])) {	    
      if (i == AVRBSLREJ) {
        if (!strstr(line, "OFF")) 
		p->isbslrejection = 1;	         
        	
      }
      else if(i == AVRREJ) {
	if (!strstr(line, "OFF")) 
		p->isrejection = 1;     	 
        
      }
      
      return i;
    }  
  }
  return AVRERROR;
}

int avrcond(char *line, int mode, AverageParameters *p)
{
  int i, occ;
  int keymode;
  char *nc, *oc, c;
  avrcondition_t *pc;
  asciiline_t  linebak;
  
  /* cond section has variable length - look for end */
  
  strcpy(linebak, line);
  cfg_line_norm(linebak);
  
  if (((keymode = avrkey(linebak, mode, p)) != AVRERROR))
    return keymode;

  /* no end - new condition */
  p->condtbl = (avrcondition_t *)
    v_realloc(p->condtbl, (p->condc+1) * sizeof(avrcondition_t), "condtbl");
  pc = &(p->condtbl[p->condc]);
  pc->codec = 0;
  pc->codetbl = NULL;

  /* new code assignment */
  if ((nc = strchr(line, '='))) {
    
    /* check for valid new code and save*/
    if (nc == 0 || nc[1] == '\0')         /* '=' is first or last char - error */
      return AVRERROR;
    nc++;
    strncpy(pc->code, nc, TRG_CODE_LENGTH);
    pc->code[TRG_CODE_LENGTH] = '\0';
    
    /* count old codes */
    i = 0;
    occ = 0;
    do {
      c = line[i];
      if (c == ',' || c == '=') {
        line[i] = 0;
        occ++;
      }
      i++;
    } while (c != '=');
    
    /* create table of old codes */
    pc->codetbl = (trgcode_t *)
      v_calloc(occ, sizeof(trgcode_t), "codetbl");

    oc = line;
    for (i = 0; i < occ; i++) {
      strncpy(pc->codetbl[i], oc, TRG_CODE_LENGTH);
      pc->codetbl[i][TRG_CODE_LENGTH] = '\0';
      oc += strlen(oc) + 1;
    }
    pc->codec = occ;
  }

  /* no assignment - code copy */
  else {
    pc->codetbl = (trgcode_t *)
      v_malloc(1 * sizeof(trgcode_t), "codetbl");

    strncpy(pc->codetbl[0], line, TRG_CODE_LENGTH);
    pc->codetbl[0][TRG_CODE_LENGTH] = '\0';
    strncpy(pc->code, line, TRG_CODE_LENGTH);
    pc->code[TRG_CODE_LENGTH] = '\0';
    pc->codec = 1;
  }

  return AVRCONDLAB;
}

int avrcondlab(char *line, int mode, AverageParameters *p)
{
  int i, l;
  char *colorstr = NULL;
  int color;
  avrcondition_t *pc;
  asciiline_t linebak;
  
  pc = &(p->condtbl[p->condc]);
  strcpy(linebak, line);

  l = strlen(linebak);
  for (i = 0; i < l; i++) { 
    if (linebak[i] == '(') {
      linebak[i]= '\0';
      colorstr = &linebak[i+1];
      if (colorstr[0] == '\0')
        return AVRERROR;
      
    }
    if (linebak[i] == ')') 
      linebak[i] = '\0';
  }
  if (colorstr == NULL || strlen(colorstr) == 0)
    return AVRERROR;
  color = cfg_get_eepcolor(colorstr);
  if (color == -1)
    return AVRERROR;

  strncpy(pc->lab, linebak, 10);
  pc->lab[10] = '\0';
  sprintf(pc->col, "color:%d", color);
  (p->condc)++;

  return AVRCOND;
}

int avrchan(char *line, int mode, AverageParameters *p, eeg_t *e)
{
  int chan;
  int chanc = eep_get_chanc(e);
  
  /* valid channel label ? - note it in table*/
  for (chan = 0; chan < chanc; chan++) {
    if (!strcasecmp(eep_get_chan_label(e, chan), line)) {
      if (p->chanc == chanc)
        return AVRERROR;
      else {
        p->chantbl[p->chanc] = chan;
        p->chanc++;
        return AVRCHAN;
      }
    }
  }
  
  /* no label - maybe key ? */
  return avrkey(line, mode, p);
}

int avrwin(char *line, int mode, AverageParameters *p, eeg_t *e)
{
  if (!cfg_get_msecint(line, &p->window, e)) {
    p->iswindow = 1;
    return AVRKEY;
  }
  else {
    return AVRERROR;
  }
}

int avrbase(char *line, int mode, AverageParameters *p, eeg_t *e)
{
  if (!strcmp(line, "ABS")) {
    p->isbaseline = 0;
    return AVRKEY;
  }
  else if (!cfg_get_msecint(line, &p->baseline, e)) {
    p->isbaseline = 1;
    return AVRKEY;
  }
  else {
    return AVRERROR;
  }
}

int avrrej(char *line, int mode, AverageParameters *p, eeg_t *e)
{ 
  if (!strcmp(line, "STANDARD") || !strcmp(line, "STD")) {
    p->rejection.start = p->window.start;
    p->rejection.length = p->window.length;
    return AVRKEY;
  }
  if (!cfg_get_msecint(line, &p->rejection, e)) {  
    return AVRKEY;
  }
  else {
    return AVRERROR;
  }
}


int avrbslrej(char *line, int mode, AverageParameters *p, eeg_t *e)
{
  if (!strcmp(line, "STANDARD") || !strcmp(line, "STD")) {
    p->bslrejection.start = p->baseline.start;
    p->bslrejection.length = p->baseline.length;
    return AVRKEY;
  }
  if (!cfg_get_msecint(line, &p->bslrejection, e)) {	  	  	  
    return AVRKEY;
  }
  else {
    return AVRERROR;
  }
}

/*
  one line parsers for unsystematic config elements
*/

int avrrefdisp(char *line, AverageParameters *p)
{
  short val;
  char *valstr = strchr(line, ':');
  
  if (!valstr || !*(valstr+1))
    return AVRERROR;
  
  if (!sscanf(valstr + 1, "%hd", &val))
    return AVRERROR;
  
  if (val == 0)
    return AVRERROR;
  
  p->refdisp = val;
  return AVRKEY;
}
  
int avrreftrg(char *line, AverageParameters *p)
{
  int i;
  size_t eaten;
  char *valstr = strchr(line, ':');
  
  
  if (p->refdisp == 0)
    return AVRERROR;
  
  if (!valstr || !*(valstr+1))
    return AVRERROR;
  else valstr++;  /* skip ':' */ 

  if (p->condc <= 0)
    return AVRERROR;
  p->reftrgv = (trgcode_t *) v_malloc(p->condc * sizeof(trgcode_t), "reftrg");
  
  valstr += strspn(valstr," \t"); /* skip any leading spaces or tabs */
  i = -1;
  while (*valstr != '\n' && *valstr != '\0' && *valstr != ';' 
        && i < p->condc) {
    if (*valstr==',') 
      return AVRERROR;        /* reject leading or multiple comma */         
    eaten = strcspn(valstr, ", \t\n"); 
    if (eaten == 0) 
      return AVRERROR;       
    p->reftrgv[++i][0] = '\0';
    strncpy(p->reftrgv[i],valstr,eaten);   
    p->reftrgv[i][eaten]='\0';
  /*  for test only
    printf("i: %d, reftrg: %-8s (eaten: %d)\n", i, p->reftrgv[i],eaten);
  */  
    valstr += eaten;
    valstr += strspn(valstr, " \t");  /* skip spaces and tabs */
    if (*valstr == ',') 
      valstr += 1 + strspn(valstr + 1, " \t"); /* skip a single comma and the following spaces or tabs */
  }
  if (i == 0) {   
    for (i = 1; i < p->condc; i++)
      strcpy(p->reftrgv[i],p->reftrgv[0]);  
  }
  if (i == -1) return AVRERROR;
  
  return AVRKEY;
}


/* initialize avr configuraion with safe values, 
   alloc memory for the channel table, abort program if allocation failes */
void InitAverageParameters( eeg_t *EEG_p, AverageParameters *p ) {	
  	
  p->condc = 0;
  p->condtbl = NULL;
  p->chanc = 0;
  p->chantbl = (short *) 
    v_calloc(eep_get_chanc(EEG_p), sizeof(short *), "chantbl"); 
  /* if v_calloc failes, eeperror will be called automatically 
     the program will be aborted 
     -> no pointer check needed here */   
  p->iswindow = 0;
  p->window.start = 0; p->window.length = 0;
  p->isbaseline = 0;
  p->baseline.start = 0; p->baseline.length = 0;
  p->isrejection = 0;
  p->rejection.start = 0; p->rejection.length = 0;
  p->isbslrejection = 0;
  p->bslrejection.start = 0; p->bslrejection.length = 0;
  
  p->refdisp = 0;
  p->reftrgv = NULL;
  
}


/* config parser main loop ----------------------------------------- */
int ReadAverageParameters(FILE *Cfg, eeg_t *EEG_p, AverageParameters *p)
{
  int i, c;
  int linenr = 0;
  int mode = AVRKEY;
  asciiline_t line, linebak, debug;
  
  /* init config items with defaults */
  InitAverageParameters(EEG_p, p);

  
  /* scan cfg file */
  do {
    if (mode == AVRERROR) {
      eepstatus("syntax error in configuration file line %d!\n\"%s\"\n",
                  linenr, line  );
      return linenr;
    }
    else {
      asciiread(Cfg, line);
      linenr++;
      if (!feof(Cfg)) {
        
        /* there are elements we have to handle in a totally different way :-( */
        c = 0;
        while(line[c]) {
          linebak[c] = toupper(line[c]);
          if (line[c] == ';')
            linebak[c] = '\0';
          c++;
        }
        linebak[c] = '\0';

        if (strstr(linebak, "DISPLACE")) {
          mode = avrrefdisp(line, p);
        }
        else if (strstr(linebak, "SCANNING")) {
          mode = avrreftrg(line, p);
        }

        else {
          if (mode == AVRCONDLAB || mode == AVRCOND)
            cfg_line_norm_cs(line);
          else
            cfg_line_norm(line);

          if (line[0]) {
            sprintf(debug, "-- %d\t%s\n", mode, line);
            eepdebug(debug);
            mode = (*avrfunc[mode]) (line, mode, p, EEG_p);
          }
        }
      }
      else {
        if (mode != AVRKEY) {
          eepstatus("unexpected end of configuration file!\n", linenr);
          return linenr;
        }
      }
    }
  } while(!feof(Cfg));

  if (!p->chanc) {
    p->chanc = eep_get_chanc(EEG_p);
    for (i = 0; i < p->chanc; i++)
      p->chantbl[i] = i;
  }

  
  if( !p->isbaseline && p->isbslrejection ) {
	  eepstatus("baseline window required for bsl rejection mode!\n"); 
          return linenr;
  }	   
  
  return 0;
}

int check_reject_window_settings(AverageParameters p)
{  
   slen_t wstart = p.window.start;
   slen_t wstop  = p.window.start + p.window.length;
   slen_t rstart = p.rejection.start;
   slen_t rstop  = p.rejection.start + p.rejection.length;
   int status = 0;

   if(!p.isbaseline) {
     /* check settings for averaging trial rejection search */  
     if ( (rstart > wstart) || (rstop < wstop) ) {     
         eeplog("####Warning: averaging window exceeds rejection window!\n");
	 status = 1;       
      }         
      return status;
    } 
    else {
     
      slen_t bstart =  p.isbaseline ? p.baseline.start : 0;
      slen_t bstop  =  p.isbaseline ? p.baseline.start + p.baseline.length : 0;
      slen_t brstart = p.isbslrejection ? p.bslrejection.start : 0;
      slen_t brstop  = p.isbslrejection ? p.bslrejection.start + p.bslrejection.length : 0;
           
     if (p.refdisp == 0) { /* without reference displacement */               
       /* do rejection windows overlap? */
       if ( rstart < brstop )
         rstart = (rstart > brstart ? brstart : rstart);
       if ( brstart < rstop )
         brstart = (brstart > rstart ? rstart : brstart);
       /* check settings for averaging trial rejection search */
       if ( ( (brstart > wstart) || (brstop < wstop) )
         && ( (rstart > wstart)  || (rstop < wstop)  )
	  ) {
	   eeplog("####Warning: averaging window exceeds rejection window!\n");
	   status = 1;
	}
       /* check settings for baseline rejection search */	     	   
       if ( ( (brstart > bstart) || (brstop < bstop) )
         && ( (rstart > bstart)  || (rstop < bstop)  )
          ) {
	    eeplog("####Warning: baseline  window exceeds rejection window!\n");
            status = 1;
	}
	return status;
      } 
      else { /* with reference displacement */ 
        /* check settings for avr-window rejection search */
	if (!p.isrejection ) {
	  eeplog("####Warning: no rejection check in averaging window!\n");
	  status = 1; 
	 }
	 else if ( (rstart > wstart) || (rstop < wstop) ) {
	   eeplog("####Warning: averaging window exceeds rejection window!\n");
	   status = 1; 
	  } 
	/* check settings for baseline rejection search */          
	if ( !p.isbslrejection ) {
	  eeplog("####Warning: no baseline rejection check!\n");
	  status = 1;
	  return status;
	 } 
        if ( (brstart > bstart) || (brstop < bstop) ) {    		  
	  eeplog("####Warning: baseline  window exceeds bsl rejection window!\n"); 
	  status = 1; 
	 }
        return status;
      }	   
    }
        
 }   	    


char *strGetConditionTriggers( AverageParameters *cfg, short condI ) {
	avrcondition_t avrCond = (*cfg).condtbl[condI];
	char *condTriggers = v_strnew(avrCond.codetbl[0],1); 
	int i;
	
	for(i=1; i < avrCond.codec; i++) 
	 condTriggers = v_strcat(strcat(condTriggers, " "), avrCond.codetbl[i],1);
	
	return condTriggers;
}	
     
 
char *strGetReferenceTrigger( AverageParameters *cfg, short condI ) {
	if( (*cfg).reftrgv ) {
	    char *refTriggers = ((*cfg).refdisp) > 0 ? 
		v_strnew("(next) ", TRG_CODE_LENGTH) : v_strnew("(previous) ", TRG_CODE_LENGTH);
			      
	    strcat(refTriggers, (*cfg).reftrgv[condI]);
	    
	    return refTriggers;
         }  	
	 
	 if( (*cfg).refdisp ) {
	    char *refTriggers = v_strnew("", 10);
	    
	    sprintf(refTriggers, "(%+d)", (*cfg).refdisp); 
	    
	    return refTriggers;
          }
	 
	  /* reference undisplaced refTriggers = condTriggers 
	     maybe we should return condTriggers here ??? */
	 return NULL;
	    	 
}

char *strGetAllReferenceTriggers( AverageParameters *cfg ) {
     if( (*cfg).reftrgv ) {
	    int condI; 
	    char *refTriggers = ((*cfg).refdisp) > 0 ? 
		v_strnew("(next)", 1) : v_strnew("(previous)", 1);
	    
	    for(condI=0; condI < (*cfg).condc; condI++) {		      
	      refTriggers = 
	        v_strcat(strcat(refTriggers," "), (*cfg).reftrgv[condI],1);
             }
	     
	    return refTriggers;
         }  	
	 
	 if( (*cfg).refdisp ) {
	    char *refTriggers = v_strnew("", 10);
	    
	    sprintf(refTriggers, "(%+d)", (*cfg).refdisp); 
	    
	    return refTriggers;
          }
	 
	  /* reference undisplaced refTriggers = condTriggers 
	     maybe we should return condTriggers here ??? */
	 return NULL;
	    	 
}

	

char *strGetAverageWindow( eeg_t *eeg, AverageParameters *cfg ) {
	double period = eep_get_period(eeg) * 1000;
	char *avrWindow = v_strnew("",80);
	
	sprintf(avrWindow, "%g..%g", (*cfg).window.start * period,
	      ((*cfg).window.start + (*cfg).window.length - 1) *period );
	return avrWindow;
}

char *strGetBaselineWindow( eeg_t *eeg, AverageParameters *cfg ) {
    if( (*cfg).isbaseline && ((*cfg).baseline.length > 0) ) {	
	
	double period = eep_get_period(eeg) * 1000;
	char *bslWindow = v_strnew("",80);
	
	sprintf(bslWindow, "%g..%g", (*cfg).baseline.start * period,
	      ((*cfg).baseline.start + (*cfg).baseline.length - 1) *period );
	return bslWindow;
    }
    
    return NULL;	
}
	 			
char *strGetRejectionWindow( eeg_t *eeg, AverageParameters *cfg ) {
     if( (*cfg).isrejection && ((*cfg).rejection.length > 0)) {	
	double period = eep_get_period(eeg) * 1000;
	char *rejWindow = v_strnew("",80);
	
	sprintf(rejWindow, "%g..%g", (*cfg).rejection.start * period,
	      ((*cfg).rejection.start + (*cfg).rejection.length - 1) *period );
	return rejWindow;
      }
      
      return NULL;	
}

char *strGetBslRejectionWindow( eeg_t *eeg, AverageParameters *cfg ) {
     if( (*cfg).isbaseline && (*cfg).isbslrejection && ((*cfg).bslrejection.length > 0) ) {	
	double period = eep_get_period(eeg) * 1000;
	char *bslrejWindow = v_strnew("",80);
	
	sprintf(bslrejWindow, "%g..%g", (*cfg).bslrejection.start * period,
	      ((*cfg).bslrejection.start + (*cfg).bslrejection.length - 1) *period );
	return bslrejWindow;
      }
      
      return NULL;	
}				

void ShowAverageParameters(AverageParameters p, eeg_t *src, short mode)
{  
  int i, j; 
  int chan;
  char *codestr;
  int color;   
  char colorstr[11]; 
  char code[sizeof(trgcode_t)]; 
  avrcondition_t *pc; 
  char refstr[] = "\n    relative to bsl reference trigger";
  char *tmp;
  char messbuf[2048];
                     
  short ignore_rejection = mode & 0x000f;
  short ignore_avrwindow = mode & 0x00f0;
  
  if (!p.refdisp) 
    refstr[0] = '\0';     
                                    
  if (!p.condc)
    eeperror("no conditions specified!\n");
  if (!p.chanc) 
    eeperror("no channels specified!\n"); 
  if (!ignore_avrwindow && !p.iswindow) 
    eeperror("averaging window required!\n"); 
  /*                                         
  if (!p.isbaseline) 
    eeperror("baseline window required!\n");
  */                                       
                                                                                 
  eeplog("trial definitions:\n");
  eeplog(" conditions:\n"); 
  for (i = 0; i < p.condc; i++) {   
    pc = &(p.condtbl)[i];
    codestr = (char *) 
      v_malloc( pc->codec * sizeof(trgcode_t) + 1, "codestr");  
    strcpy(codestr, ""); 
    for (j = 0; j < pc->codec; j++) {   
      sprintf(code, "%s ", pc->codetbl[j]); 
      strcat(codestr, code); 
    }                       
    sscanf(pc->col + 6, "%d", &color);
    sprintf(messbuf, "  %-10s %s= %2s   (%s)\n",
      pc->lab, 
      codestr, 
      pc->code,
      cfg_put_eepcolorstr(color, colorstr)); 
    eeplog(messbuf);
    v_free(codestr); 
  } 
                
  eeplog(" channels:\n  ");  
  if (p.chanc == eep_get_chanc(src)) { 
    sprintf(messbuf, "all channels (%d)\n", p.chanc);
    eeplog(messbuf);
  }
  else {  
    for (i = 0; i < p.chanc; i++) {  
      chan = p.chantbl[i];  
      sprintf(messbuf, "%s ", eep_get_chan_label(src, chan));  
      eeplog(messbuf); 
    } 
    eeplog("\n");  
  }
  
  eeplog(" windows:\n");
  if(!ignore_avrwindow) {
    sprintf(messbuf, "  averaging window:  (ms) %s\n",
          tmp = strGetAverageWindow( src, &p ) );
    v_free(tmp);		      
    eeplog(messbuf);  
   }  
                    
  if (p.isbaseline) {
    sprintf(messbuf, "  baseline window:   (ms) %s\n",
	tmp=strGetBaselineWindow( src, &p ) );
    v_free(tmp);	          
    eeplog(messbuf); 
    if (p.refdisp) {
      if (!p.reftrgv) {  
        sprintf(messbuf, "    relative to trigger %+d\n", p.refdisp); 
        eeplog(messbuf);
      } 
      else {          
	eeplog("    relative to (one code for each condition)\n      "); 
	eeplog(tmp=strGetAllReferenceTriggers( &p )); 
	v_free(tmp);	
        eeplog("\n");  
      }
    }  
  }
  else {
    eeplog("  no baseline calculation\n");
  }
  
  if ((p.isrejection || p.isbslrejection) && !ignore_rejection) { 
    if(p.rejection.length > 0) {
      eeplog("  rejection window:  (ms) %s\n",
	 tmp=strGetRejectionWindow( src, &p ) );
      v_free(tmp);	            
     }   
    if(p.isbaseline && p.isbslrejection) {
      eeplog("  bsl reject window: (ms) %s %s\n",
	 tmp=strGetBslRejectionWindow( src, &p ), refstr );
      v_free(tmp);	               
     }
    if(!ignore_avrwindow) { 	
      if (check_reject_window_settings(p)) 
        eeplog("####Check settings in the configuration file!\n"); 
    }     
  }
  else {
   if( !ignore_rejection )	  
    eeplog("####Warning:  no rejection check!\n");  
  } 
} 
			  
#ifdef OLDSHOW 
void ShowAverageParameters(AverageParameters p, eeg_t *src)
{  
  int i, j; 
  int chan;
  char *codestr;
  int color;   
  char colorstr[11]; 
  char code[4]; 
  avrcondition_t *pc; 
  char refstr[] = "(relative to bsl reference trigger)";
                     
  float period = eep_get_period(src);
  
  if (!p.refdisp) 
    refstr[0] = '\0';     
                                    
  if (!p.condc)
    eeperror("no conditions specified!\n");
  if (!p.chanc) 
    eeperror("no channels specified!\n"); 
  if (!p.iswindow) 
    eeperror("averaging window required!\n"); 
  /*                                         
  if (!p.isbaseline) 
    eeperror("baseline window required!\n");
  */                                       
                                                                                 
  eeplog("average parameters:\n");
  eeplog(" conditions:\n"); 
  for (i = 0; i < p.condc; i++) {   
    pc = &(p.condtbl)[i];
    codestr = (char *) 
      v_malloc( pc->codec * sizeof(trgcode_t) + 1, "codestr");  
    strcpy(codestr, ""); 
    for (j = 0; j < pc->codec; j++) {   
      sprintf(code, "%s ", pc->codetbl[j]); 
      strcat(codestr, code); 
    }                       
    sscanf(pc->col + 6, "%d", &color);
    sprintf(messbuf, "  %-10s %s= %2s   (%s)\n",
      pc->lab, 
      codestr, 
      pc->code,
      cfg_put_eepcolorstr(color, colorstr)); 
    eeplog(messbuf);
    v_free(codestr); 
  } 
                
  eeplog(" channels:\n  ");  
  if (p.chanc == eep_get_chanc(src)) { 
    sprintf(messbuf, "all channels (%d)\n", p.chanc);
    eeplog(messbuf);
  }
  else {  
    for (i = 0; i < p.chanc; i++) {  
      chan = p.chantbl[i];  
      sprintf(messbuf, "%s ", eep_get_chan_label(src, chan));  
      eeplog(messbuf); 
    } 
    eeplog("\n");  
  }
  
  eeplog(" windows:\n");
  sprintf(messbuf, "  averaging window:  (ms) %.0f..%.0f\n",
    p.window.start * period * 1000.0,
    (p.window.start + p.window.length - 1) * period * 1000.0  
  );
  eeplog(messbuf);  
                    
  if (p.isbaseline) {
    sprintf(messbuf, "  baseline window:   (ms) %.0f..%.0f\n",
      p.baseline.start * period * 1000.0,
      (p.baseline.start + p.baseline.length - 1) * period * 1000.0 
    ); 
    eeplog(messbuf); 
    if (p.refdisp) {
      if (!p.reftrgv) {  
        sprintf(messbuf, "    relative to trigger %+d\n", p.refdisp); 
        eeplog(messbuf);
      } 
      else {  
        if (p.refdisp > 0) {  
          eeplog("    relative to next (one code for each condition)\n      ");
        } 
        else { 
          eeplog("    relative to previous (one code for each condition)\n      "); 
        }
        for (i = 0; i < p.condc; i++) { 
          sprintf(messbuf, "%s ", p.reftrgv[i]); 
          eeplog(messbuf);
        }
        eeplog("\n");  
      }
    }
  }
  else {
    eeplog("  no baseline calculation\n");
  }
  
  if (p.isrejection) { 
    if(p.rejection.length > 0) {
      eeplog("  rejection window:  (ms) %.0f..%.0f\n",
      p.rejection.start * period * 1000.0, 
      (p.rejection.start + p.rejection.length - 1) * period * 1000.0); 
     } 
    if(p.isbaseline && p.isbslrejection) {
      eeplog("  bsl reject window: (ms) %.0f..%.0f %s\n",
         p.bslrejection.start * period * 1000.0,
	 (p.bslrejection.start + p.bslrejection.length - 1) * period * 1000.0,
	 refstr);
     }	
    if (check_reject_window_settings(p)) {
      eeplog("####Check settings in the configuration file!\n"); 
    }     
  }
  else {
    eeplog("####Warning:  no rejection check!\n");  
  } 
} 
#endif


void FreeAverageParameters(AverageParameters *p)
{
  int i;
  avrcondition_t *pc;
  
  for (i = 0; i < p->condc; i++) {
    pc = &p->condtbl[i];
    v_free(pc->codetbl);
  }
  v_free(p->condtbl); p->condc = 0;
  v_free(p->chantbl); p->chanc = 0;
  p->iswindow = p->isbaseline = p->isrejection = 0; p->isbslrejection = 0;
}

#endif /* ! defined(WIN32) || defined(__CYGWIN__) */
