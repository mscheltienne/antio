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

#include <avr/avr.h>
#include <eep/eepio.h>
#include <eep/eepmem.h>
#include <eep/eepraw.h>
#include <eep/winsafe.h>

#ifdef COMPILE_RCS
char RCS_avr_h[] = RCS_AVR_H;
char RCS_avr_c[] = "$RCSfile: avr.c,v $ $Revision: 2415 $";
#endif

/* avr files --------------------------------------------------------*/

#ifdef NEVER_COMPILE_THIS
/* Pseudo-structure for .AVR-files */

/* int   = 2 Bytes
   long  = 4 Bytes
   float = 4 Bytes
   char arrays NOT null terminated */

struct LocalHeader {
  char          ChanLabel[10];                  /* channel label */
  long          fPtr;                           /* offset of data in file */
  char          dummy[2];                       /* reserved */
};

struct GlobalHeader {
  int           GlobalHeaderSize;               /* ..currently always 38 */
  int           LocalHeaderSize;                /* ..currently always 16 */
  int           NCHANNELS;                      /* number of channels */
  int           NSAMPLES;                       /* number of samples */
  int           nTrials;                        /* total number of trials */
  int           nRejected;                      /* number of rejected trials */
  float         msLeft;        /* ms-value corresponding to first data point */
  float         SampleIntervall;                /* sampling intervall in ms */
  char          CondLabel[10];                  /* condition label */
  char          CondColor[8];                   /* associated color */
  
  
};

struct ChanData {
  float         means[NSAMPLES];                /* average data */
  float         variances[NSAMPLES];            /* corresponding variances */
};

struct AVR_file {
  struct GlobalHeader  Header;                  /* variable size! */
  struct LocalHeader ChanInfo[NCHANNELS];       /* channel info table,
                   size depends on dynamically determined value of NCHANNELS */
  char    History[NHistoryLines];               /* file-history (ascii), 
                                    enclosed between "[History]"  und "EOH\n"*/
  struct ChanData      Data[NCHANNELS];
};
#endif

/*
  avr file format offsets
*/

/* #define AVRHEADER_SIZE 38 */

#define AVROFS_HEADER_SIZE 0
#define AVROFS_CHANNEL_HEADER_SIZE 2

#define AVROFS_CHANC 4
#define AVROFS_SAMPLEC 6
#define AVROFS_TRIALC 8
#define AVROFS_REJTRIALC 10
#define AVROFS_STARTTIME 12
#define AVROFS_PERIOD 16
#define AVROFS_CONDLAB 20
#define AVROFS_CONDCOL 30

/* #define AVRCHANNELHEADER_SIZE 16 */
#define AVROFS_LAB 0
#define AVROFS_FILEPOS 10


#define AVROFS_HISTORY(x) ( (x)->header_size + (x)->chanc * (x)->channel_header_size )
#define AVRHIST_TAG "[History]"
#define AVRHIST_END "EOH\n"


void init_avr_history( avr_t *avr ) {
    
    avr->histc = 0;
    avr->hist_size = 0;
    avr->histv = NULL;
}   


int append_avr_history( avr_t *avr, const char *line) {
  size_t len = strlen(line);
  
  if(len == 0) 
      return -1;
      
  if(avr->histc == 0) 
      avr->histv = (char **) v_malloc( sizeof(char *), "histv");
   else 
      avr->histv = (char **) v_realloc(avr->histv, (avr->histc + 1) * sizeof(char *), "histv");
       
   avr->histv[avr->histc] = v_strnew(line, 0);
   avr->histc++;
   avr->hist_size += len;
   
   return 0;
	  
}

int get_avr_history(avr_t *avr, FILE *f) { 
 
 char line[128] = "";
 char *got = NULL;
 unsigned short notEOH = 1;
 
  init_avr_history( avr );
  eepio_fseek(f, AVROFS_HISTORY(avr), SEEK_SET);
  got = fgets(line, strlen(AVRHIST_TAG) + 1, f); 
  
  if( got && !strncmp(line,AVRHIST_TAG,9) ) {
      avr->hist_size += strlen(line);	  
      fgets(line, 128, f); /* maybe later we want to log the histsize after the tag?? */	            
      avr->hist_size += strlen(line); 
      while( !feof(f) && got && notEOH ) { 
           got =  fgets(line, 128, f);
	   if( got && (notEOH = strncmp(line, AVRHIST_END, 4)) )  
               append_avr_history( avr, line );
	    
       }
      if( !got || notEOH ) 
         eeperror("error reading avr file history\n");	
       else avr->hist_size += strlen(line); 
	  	 	    
   }
    
  return ferror(f);
}

int put_avr_history( avr_t *avr, FILE *f ) {
  
  int i;
      
  if( avr->histc > 0 ) {      
     eepio_fseek(f, AVROFS_HISTORY(avr), SEEK_SET);
  
     fputs(AVRHIST_TAG, f);
     fputs("\n", f); /* maybe, later we'll log the histsize after the tag */
     avr->hist_size = strlen(AVRHIST_TAG) + 1; /* depends on the tag line! */
     for(i=0; i< avr->histc; i++) { 
         fputs(avr->histv[i], f);
         avr->hist_size += strlen(avr->histv[i]);
      } 
           
     fputs(AVRHIST_END, f);
     avr->hist_size += strlen(AVRHIST_END);
  }
  
  return ferror(f); 
} 


void copy_avr_history( avr_t *src, avr_t *dst) {
  
  int i;	
	
  dst->hist_size = src->hist_size;
  dst->histc = src->histc;
  if(dst->histc) {
     dst->histv = (char **) v_malloc(dst->histc * sizeof(char *), "histv");
     for(i=0; i<dst->histc; i++) 
        dst->histv[i] = v_strnew( src->histv[i], 0 );
   } else {	
        dst->histv = NULL;
	
   }	   

}

void show_avr_history( avr_t *avr, int linelen ) {
	
   /* write avr history to stdout, wrap lines after linelen chars */
	
   int i, len, n, c, lineleft = linelen;
   char *str;
      
   if( avr->histc <= 0 ) {
	   fprintf(stdout, "  none available\n");
           return; 
    }	   
   
   for( i=0; i < avr->histc; i++ ) {
	   str = avr->histv[i];
	   c = 0; 
	   len = strlen(str); 	   
	   while( len ) {
		   /* avoid empty lines */
		   if( str[0] == '\n')  { str++; len--; }
		   if( len == 0 ) continue;
		   n = (len > lineleft ? lineleft : len);
		   c = eepio_fwrite(str, sizeof(char), n, stdout);
		   if( c != n) 
                       syserror("output error"); 
	           str += c; 
		   len = strlen(str);
	           /* how many characters left on the line ? */  
		   if( str[-1] == '\n' ) {
			   lineleft = linelen;
		    } else if(len) {
                           eepio_fwrite("\n",sizeof(char),1,stdout);
			   lineleft = linelen;			
		    } else { 
                          lineleft = lineleft > c ? lineleft - c : linelen;
		    }	  
		    	  			     		    		    
			 			                    
           }
   }
   
   fputs("\n",stdout);  
}   
	


void free_avr_history( avr_t *avr ) {
   int i;
   
   if(avr->histc > 0) {
	   for(i=0; i<avr->histc; i++)
		   v_free(avr->histv[i]);
           v_free(avr->histv);
	   
   } 
   
   init_avr_history( avr ); /* initialize with safe values */
      
}  	   	   		


short get_avr_headerSize(avr_t *avr) {
	return avr->header_size;
}

short get_avr_channelHeaderSize(avr_t *avr) {
	return avr->channel_header_size;
}

size_t get_avr_histSize(avr_t *avr) {
	return avr->hist_size;
}					

size_t get_avr_totalHeaderSize(avr_t *avr) {
	return (size_t)  avr->header_size 
		       + avr->channel_header_size * avr->chanc
		       + avr->hist_size;
}			

int get_avr_header(avr_t *avr, FILE *f)
{
  int i;
  float t = 0.0, p = 0.0;
  int v;
  int w;
  
  /* load the format information */
  eepio_fseek(f, AVROFS_HEADER_SIZE, SEEK_SET);
  read_u16(f, &w);
  avr->header_size = w;
  eepio_fseek(f, AVROFS_CHANNEL_HEADER_SIZE, SEEK_SET);
  read_u16(f, &w);
  avr->channel_header_size = w;
  
  eepio_fseek(f, AVROFS_CHANC, SEEK_SET);
  read_u16(f, &v);
  avr->chanc = v;
  
  eepio_fseek(f, AVROFS_SAMPLEC, SEEK_SET);
  read_u16(f, &v);
  avr->samplec = v;

  eepio_fseek(f, AVROFS_TRIALC, SEEK_SET);
  read_u16(f, &v);
  avr->trialc = v;

  eepio_fseek(f, AVROFS_REJTRIALC, SEEK_SET);
  read_u16(f, &v);
  avr->rejtrialc = v;
  
  eepio_fseek(f, AVROFS_STARTTIME, SEEK_SET);
  read_f32(f, &t);
  if (t > 1E6 || t < -1E6) return 1;
  
  eepio_fseek(f, AVROFS_PERIOD, SEEK_SET);
  read_f32(f, &p);
  if (p <= 0.0001 || p > 1E6) return 1;
  
  avr->period = (float) (p / 1000.0);
  avr->sample0 = (int) FRND(t / p);
  
  /* condition label has trailing spaces - ignore them */
  eepio_fseek(f, AVROFS_CONDLAB, SEEK_SET);
  i = 0;
  while (i < 11 && (avr->condlab[i++] = fgetc(f)) != ' ');
  avr->condlab[i - 1] = '\0';

  /* color code uses 8 chars always */
  eepio_fseek(f, AVROFS_CONDCOL, SEEK_SET);
  eepio_fread(avr->condcol, 8, 1, f);
  avr->condcol[8] = '\0';
  
  avr->mtrialc = (float) (avr->trialc - avr->rejtrialc);
  
  

  return ferror(f);
}

int get_chan_header(avr_t *avr, FILE *f, short chan)
{
  int i;
  uint32_t u32;
  const int ofs = avr->header_size + chan * avr->channel_header_size;
  avrchan_t *c = &(avr->chanv[chan]);
  
  eepio_fseek(f, ofs + AVROFS_LAB, SEEK_SET); 
  i = 0;
  while (i < 11 && (c->lab[i++] = fgetc(f)) != ' ');
  c->lab[i - 1] = '\0';
  
  eepio_fseek(f, ofs + AVROFS_FILEPOS, SEEK_SET);
  read_u32(f, &u32);
  c->filepos = u32;
  
  return ferror(f);
}

int put_avr_header(avr_t *avr, FILE *f)
{
  eepio_fseek(f, 0, SEEK_SET);
  write_u16(f, avr->header_size);
  write_u16(f, avr->channel_header_size);
  eepio_fseek(f, AVROFS_CHANC, SEEK_SET);
  write_u16(f, avr->chanc);
  eepio_fseek(f, AVROFS_SAMPLEC, SEEK_SET);
  write_u16(f, avr->samplec);
  eepio_fseek(f, AVROFS_TRIALC, SEEK_SET);
  write_u16(f, avr->trialc);
  eepio_fseek(f, AVROFS_REJTRIALC, SEEK_SET);
  write_u16(f, avr->rejtrialc);
  eepio_fseek(f, AVROFS_STARTTIME, SEEK_SET);
  write_f32(f, (float) avr->sample0 * avr->period * 1000);
  eepio_fseek(f, AVROFS_PERIOD, SEEK_SET);
  write_f32(f, avr->period * 1000);
  
  eepio_fseek(f, AVROFS_CONDLAB, SEEK_SET);
  avr->condlab[10] = '\0';
  fprintf(f, "%-10s", avr->condlab);       /* strings are filled with spaces! */
  eepio_fseek(f, AVROFS_CONDCOL, SEEK_SET);
  avr->condcol[8] = '\0';
  fprintf(f, "%-8s", avr->condcol);
  
  
    
  return ferror(f);
}

int put_chan_header(avr_t *avr, FILE *f, short chan)
{
  avrchan_t *c = &avr->chanv[chan];
  const unsigned int ofs = avr->header_size + chan * avr->channel_header_size;
  char dummy[2] = {'\0','\0'};
  
  
  eepio_fseek(f, ofs + AVROFS_LAB, SEEK_SET);
  c->lab[10] = '\0';
  fprintf(f, "%-10s", c->lab);
  
  /* setting the file pointer */
  eepio_fseek(f, ofs + AVROFS_FILEPOS, SEEK_SET);
  write_u32(f, c->filepos);
  
  /* insert dummy */
  eepio_fwrite(dummy,2,1,f);
  
  
  return ferror(f);
}

char *get_avr_chan_lab(avr_t *avr, short indx) {
   if (indx<0 || indx >= avr->chanc) return NULL;
   return avr->chanv[indx].lab;	
}	

unsigned short get_avr_chanc(avr_t *avr)
{
  return avr->chanc;
}  			

unsigned short get_avr_trialc(avr_t *avr)
{
  return avr->trialc;
}  			

unsigned short get_avt_rejectc(avr_t *avr)
{
  return avr->rejtrialc;
}  			

uint64_t get_avr_samplec(avr_t *avr)
{
   return avr->samplec;
}   

float get_avr_period(avr_t *avr)
{
   return avr->period;
}   						

int avropen  (avr_t *avr, FILE *f)
{
  int i;
  
  if (get_avr_header(avr, f)) return AVRERR_FILE;
  if (avr->chanc < 1 ) return AVRERR_DATA;
  
  avr->chanv = (avrchan_t *)
    v_malloc(avr->chanc * sizeof(avrchan_t), "avrchanv");
  for(i = 0; i < avr->chanc; i++) 
    if( get_chan_header(avr, f, i) ) return AVRERR_FILE;
  
  /* get file history */
  if( get_avr_history(avr, f) ) return AVRERR_FILE;
  
  return AVRERR_NONE;
}

#ifdef OBSOLETE  /* inappropriate to read avr files containing a history */
int avropen_f(avr_t *avr, FILE *f)
{
  int w;
  float t, p;
  char labbuf[16], dummy[2];
  int chan;
  
  rewind(f);
  
  /* load the format information */
  read_u16(f, &w);
  avr->header_size = w;
  read_u16(f, &w);
  avr->channel_header_size = w;
  
  /* this function depends on the default header - it doesn't eepio_fseek */
  if (avr->header_size != 38 || avr->channel_header_size != 16)
    return avropen(avr, f);
  
  /* load global inf. */
  read_u16(f, &w);
  avr->chanc = w;
  read_u16(f, &w);
  avr->samplec = w;
  read_u16(f, &w);
  avr->trialc = w;
  read_u16(f, &w);
  avr->rejtrialc = w;
  avr->mtrialc = avr->trialc - avr->rejtrialc;
  if (avr->chanc < 1 || avr->samplec < 1)
    return 1;

  read_f32(f, &t);
  if (t > 1E6 || t < -1E6) return 1;
  read_f32(f, &p);
  if (p <= 0.0001 || p > 1E6) return 1;
  avr->period = p / 1000.0;
  avr->sample0 = FRND(t / p);
  
  /* condition label has trailing spaces - ignore */
  labbuf[10] = '\0';
  eepio_fread(labbuf, 10, 1, f);
  sscanf(labbuf, "%s", avr->condlab);

  /* color code uses 8 chars always */
  eepio_fread(avr->condcol, 8, 1, f);
  avr->condcol[8] = '\0';
  
  avr->chanv = (avrchan_t *)
    v_malloc(avr->chanc * sizeof(avrchan_t), "avrchanv");
  for (chan = 0; chan < avr->chanc; chan++) {
    eepio_fread(labbuf, 10, 1, f);
    sscanf(labbuf, "%s", avr->chanv[chan].lab);
    read_s32(f, &(avr->chanv[chan].filepos));
    eepio_fread(dummy, 2, 1, f);
  }

  return ferror(f);
}
#endif

int avrclose(avr_t *avr)
{
  if( avr ) {	
      v_free(avr->chanv);
      free_avr_history( avr);
   }   
  return AVRERR_NONE;
}

int avrnew  (avr_t *avr, FILE *f, const char *registry, const char *cmdline)
{
  int i;
  
  if(registry)           
      append_avr_history(avr, registry);
  if(cmdline)
      append_avr_history(avr, cmdline);
   
  
  avr->header_size = AVR_HEADER_SIZE;
  avr->channel_header_size = AVR_CHANNEL_HEADER_SIZE;
      
  
  if ( put_avr_header(avr, f) ) return AVRERR_FILE;
  
  /* dump file history before writing the channel headers! 
     the exact history length must be known for computing the channel offsets
   */
  if ( put_avr_history(avr, f) ) return AVRERR_FILE;
  
  for (i = 0; i < avr->chanc; i++) {
     avr->chanv[i].filepos = 
      AVROFS_HISTORY( avr ) + avr->hist_size		           		     
      + i * 2 * 4 * avr->samplec;

    if ( put_chan_header(avr, f, i) ) return AVRERR_FILE;
  }    
  
  return AVRERR_NONE;
}

int avrseek (avr_t *avr, FILE *f, short chan, short band)
{
  int r;
  long offs = band * 4 * avr->samplec;
     
  r = eepio_fseek(f, (long) avr->chanv[chan].filepos + offs, SEEK_SET);
  
  return (r ? AVRERR_FILE : AVRERR_NONE);
}

int avrread (FILE *f, float *v, uint64_t c)
{
  uint64_t r;
  
  r = vread_f32(f, v, c);  
  
  return (r != c ? AVRERR_FILE : AVRERR_NONE);
}

int avrwrite(FILE *f, float *v, uint64_t c)
{
  uint64_t r;
  
  r = vwrite_f32(f, v, c);
   
  return (r != c ? AVRERR_FILE : AVRERR_NONE);
}  

void avrcopy (avr_t *src, avr_t *dst, short retain_history)
{
  memcpy(dst, src, sizeof(avr_t));
  
  if (dst->chanv) {
    dst->chanv = (avrchan_t *)
      v_malloc(dst->chanc * sizeof(avrchan_t), "avrchanv");
    memcpy(dst->chanv, src->chanv, dst->chanc * sizeof(avrchan_t));
  }
  
  if(retain_history)
          copy_avr_history( src, dst );
  else  
	  init_avr_history( dst );  	  
	  	  	  
}
	

short avr_eep_get_chan_index(avr_t *avr, char *lab, short try_first)
{
  short r; 
  
  if ( (try_first < avr->chanc) 
     && !strcasecmp(avr->chanv[try_first].lab, lab)) {
    r = try_first;
  }
  else {
    short chan = 0;
    while(chan < avr->chanc && strcasecmp(avr->chanv[chan].lab, lab)) chan++;
    r = chan < avr->chanc ? chan : -1;
  }
  return r;
}

float **avr_load(avr_t *avr, FILE *f, float **v, 
              chanlab_t *chanv, short chanc, uint64_t sample0, uint64_t samplec,
              int band)
{
  float *in = (float *) v_malloc(avr->samplec * sizeof(float), "in");
  
  short rchanc = chanc ? chanc : avr->chanc;
  short rsamplec = samplec ? samplec : avr->samplec;
  float **rv = v ? v : (float **) v_malloc_s2d(rchanc, rsamplec);
  short i, c;

  for (i = 0; i < rchanc; i++) {
    if (chanc) {
      c = avr_eep_get_chan_index(avr, chanv[i], i);
      if (c < 0)
        eeperror("avr_load: required channel not found!\n");
    }
    else {
      c = i;
    }
    if (    avrseek(avr, f, c, band)
         || avrread(f, in, avr->samplec) )
      syserror("avr_load: error reading avr!\n");
    
    memcpy(rv[i], &in[sample0 - avr->sample0], rsamplec * sizeof(float));
    /*
    for (sample = 0; sample < rsamplec; sample++) {
      rv[i][sample] = in[sample0 - avr->sample0 + sample];
    }
    */
  }
  v_free(in);

  return rv;
}

void avr_save(avr_t *avr, FILE *f, float **v, int band)
{
  short chan;
  for (chan = 0; chan < avr->chanc; chan++) {
    if (   avrseek(avr, f, chan, band)
        || avrwrite(f, v[chan], avr->samplec) )
      syserror("avr_save: error writing avr!\n");
  }
}

int avr_read_slice(avr_t *avr, FILE *Avr, uint64_t start, uint64_t length, 
                    chanlab_t *chanv, short chanc, float *slice)
{
  short chan, i;
  uint64_t sample;
  float *in;
  
  if (length == 0) length = 1;
  
  in = (float *) v_malloc(avr->samplec * sizeof(float), "in");
  
  sample = 0;

  for (chan = 0; chan < chanc; chan++) {
    if ((i = avr_eep_get_chan_index(avr, chanv[chan], chan)) < 0) {
      eeperror("channel %s not in avr\n", chanv[chan]);
    }
    if (    avrseek(avr, Avr, i, AVRBAND_MEAN)
         || avrread(Avr, in, sample + 1) )
    {
      v_free(in);
      return 1;
    }
    slice[chan] = 0;
    for (sample = start; sample < start + length; sample++) 
      slice[chan] += in[sample];
    slice[chan] /= length;
  }

  v_free(in);
  return 0;
}

int avr_var_valid(float *v, int c)
{
  return !((v[0] == 0.0 && v[c - 1] == 0.0) || (v[0] == 1.0 && v[c - 1] == 1.0));
}
