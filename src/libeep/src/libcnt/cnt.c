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

#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#include <eep/stdint.h>
#include <eep/inttypes.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <avr/avr.h>
#include <cnt/cnt_private.h>
#include <cnt/cnt_version.h>
#include <eep/eepio.h>
#include <eep/eepmem.h>
#include <eep/eepraw.h>
#include <eep/var_string.h>
#include <eep/winsafe.h>

#ifdef CNT_MMAP
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef COMPILE_RCS
char RCS_cnt_h[] = RCS_CNT_H;
char RCS_cnt_c[] = "$RCSfile: cnt.c,v $ $Revision: 2415 $";
#endif

/* used to test the binary headers */
#define CNT_MIN_PERIOD  0.00001
#define CNT_MAX_PERIOD 10.00000
#define CNT_MAX_CHANC  2048

/* EEP 2.0 / NeuroScan header definitions */
#define OFS_INFO          0  /* unused */
#define OFS_TYPE         20  /* char type =0 for SCAN 3.x cnt's,
                                          =1 for SCAN 4.1 cnt's
                              */
#define OFS_CNTTYPE    885   /* char ContinuousType
                              =3 for SCAN 3.x cnt's (blocked format),
                              =1 for SCAN 4.1 cnt's (multiplexed)
                              */

#define OFS_INFOT         30  /* unused */
#define OFS_NCHAN        370
#define OFS_RATE        376

#define OFS_EVTC        353  /* NS30 event table */
#define OFS_EVTPOS      886
#define OFS_BLOCKL      894  /* NS30 block length in bytes */

/* EEP 2.0 / NeuroScan electrode header information */
#define OFS_LAB           0
#define OFS_SENS         59   /* NS30 only */
#define OFS_CALIB         71

#define UNCOMPR_FLOAT_32 12 /* Store uncompressed FLOAT32 values */
#define RESERVED_FOR_COMPR_FLOAT_32 13 /* Store compressed FLOAT32s - not implemented */

#define TF_CBUF_SIZE(cnt, n) ( (cnt)->eep_header.chanc * (cnt)->tf_header.componentc * (n) * sizeof(float) + (cnt)->eep_header.chanc * (cnt)->tf_header.componentc)
#define FLOAT_CBUF_SIZE(cnt, n) ( (cnt)->eep_header.chanc * sizeof(float) * (n) + (cnt)->eep_header.chanc )

#define TAG_EEP20 "EEP V2.0"
#define TAG_NS30  "Version 3.0"

#define RSCALE_EEP20 (100.0 / 2048.0)  /* EEP 2.0 built-in uV/bit scaling */
#define RSCALE_NS30  ( 10.0 / 2048.0)

/* These macro's can be used to return immediately when error conditions occur.
  Use them like this:
    RET_ON_CNTERROR(function_that_returns_cnterrcode());
  And:
    RET_ONRIFFERROR(function_that_returns_rifferrcode(), SOME_CNTERR_CODE);

  (the "do .. while(0)" prevents macro inclusion problems - they will be
  optimized away by the compiler)
*/

#define RET_ON_CNTERROR(x) do { int i; if (CNTERR_NONE != (i = (x)) ) return (i); } while (0);
#define RET_ON_RIFFERROR(x, y) do { if (RIFFERR_NONE != (x) ) return (y); } while (0);

#if defined(WIN32) && !defined(__CYGWIN__)
#define NOT_IN_WINDOWS(x) do {} while(0);
#else
#define NOT_IN_WINDOWS(x) do {x;} while(0);
#endif

/* These are declarations of 'private' helper functions that are used for
  reading and writing the several chunks */

/* Helpers for reading cnt data */
int init_data_store(eeg_t *cnt, eep_datatype_e type);
int read_trigger_chunk(eeg_t *cnt);
int read_chanseq_chunk(eeg_t *cnt, storage_t *store, uint64_t expected_length);
int read_epoch_chunk(eeg_t *cnt, storage_t *store);
int read_recinfo_chunk(eeg_t *cnt, record_info_t* recinfo);

int getepoch_impl(eeg_t *cnt, eep_datatype_e type, uint64_t epoch);
int eep_seek_impl(eeg_t *cnt, eep_datatype_e type, uint64_t s, int rel);

/* Helpers for writing cnt data */
int write_eeph_chunk(eeg_t *cnt);
int write_tfh_chunk(eeg_t *cnt);
int make_partial_output_consistent(eeg_t *cnt, int finalize);
int close_data_chunk(eeg_t *cnt, int finalize, storage_t *store /*, chunk_mode_e write_mode*/);

int saveold_RAW3(eeg_t *dst, eeg_t *src, unsigned long delmask);
int puthead_EEP20(eeg_t *EEG);
int getepoch_NS30(eeg_t *EEG, uint64_t epoch);

int eep_create_file_EEP20(eeg_t *dst, eeg_t *src, unsigned long delmask);

int write_trigger_chunk(eeg_t *cnt);
int write_chanseq_chunk(eeg_t *cnt, storage_t *store, uint64_t num_chans);
int write_epoch_chunk(eeg_t *cnt, storage_t *store);
int write_recinfo_chunk(eeg_t *cnt, record_info_t* recinfo);

int putepoch_impl(eeg_t *cnt);

/* General */
int cnt_create_raw3_compr_buffer(eeg_t *cnt);
int tf_convert_for_read(eeg_t *cnt, char *in, float* out, int length);
long tf_convert_for_write(eeg_t *cnt, float *in, char* out, int length);
int rawf_convert_for_read(eeg_t *cnt, char *in, float* out, int length);
long rawf_convert_for_write(eeg_t *cnt, float *in, char* out, int length);

int decrease_chunksize(FILE* f, chunk_t* chunk, uint64_t to_subtract, int is_cnt_riff);


/* If 'key' matches 'line', read the next line from 'f' and parse it
  using scanf("%s", res).
  Returns the number of bytes read from the file (0 if there was no match).
  N.B. Unfortunately we can't make this function handle arbitrary types/format
  strings because vsscanf is not in the ANSI C standard :( */
int match_config_str(FILE *f, const char *line, const char *key, char *res, int max_len);

#ifdef CNT_MMAP
int mmap_data_chunk(FILE *f, storage_t *store);
#endif


int match_config_str(FILE *f, const char *line, const char *key, char *res, int max_len)
{
  int len = 0;
  if (strstr(line, key))
  {
    len = strlen(fgets(res, max_len, f));
    if( res[len-1] == '\n' ) res[len-1] = '\0';
  }
  return len;
}

int eep_seek_impl(eeg_t *cnt, eep_datatype_e type, uint64_t s, int rel)
{
  uint64_t newpos = s;
  storage_t *store = &cnt->store[type];
  if (!store->initialized)
    return CNTERR_DATA; /* No such data in this file */

  if (rel)
    newpos = store->data.bufepoch * store->epochs.epochl + store->data.readpos + s;

  if (newpos >= (DATATYPE_TIMEFREQ == type ? eep_get_tf_samplec(cnt) : eep_get_samplec(cnt)) ||
      newpos < 0)
    return CNTERR_RANGE;

  if (newpos  / store->epochs.epochl != store->data.bufepoch)
    RET_ON_CNTERROR(getepoch_impl(cnt, type, newpos / store->epochs.epochl));
  store->data.readpos = newpos % store->epochs.epochl;
  return CNTERR_NONE;
}
/*
void
fgets_hack(char *s, int size, FILE *f) {
  if(fgets(s,size,f)==NULL) {
    printf("warning, fgets returned NULL\n");
  }
  printf("read: %s\n", s);
  printf("strlen(%s): %i\n", s, strlen(s));
}
#define fgets fgets_hack
*/
int gethead_RAW3(eeg_t *EEG)
{
  FILE *f = EEG->f;
  int nread = 0,
      nread_last = 0;
  uint64_t nread_total;
  char line[128];
  char histline[2048];
  double rate = -1.0;
  int chan;

  if(EEG->mode==CNT_RIFF) {
    nread_total=EEG->eeph.size;
  } else {
    nread_total=EEG->eeph.size;
  }
  do {
    fgets(line, 128, f); nread += strlen(line);

    if (*line == '[') {
      if (strstr(line, "[File Version]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%d.%d", &EEG->eep_header.fileversion_major, &EEG->eep_header.fileversion_minor);
      }
      else if (strstr(line, "[Sampling Rate]")) {
        fgets(line, 128, f); nread += strlen(line);
        if (sscanf(line, "%lf", &rate) != 1 || rate < 1e-30)
          return 1;
        EEG->eep_header.period = 1.0 / rate;
      }
      else if (strstr(line, "[Samples]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%" SCNd64, &EEG->eep_header.samplec);
      }
      else if (strstr(line, "[Channels]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%hd", &EEG->eep_header.chanc);
      }
      else if (strstr(line, "[Basic Channel Data]")) {
        if (EEG->eep_header.chanc < 1)
          return 1;
        EEG->eep_header.chanv = (eegchan_t *)
          v_malloc(EEG->eep_header.chanc * sizeof(eegchan_t), "chanv");
        chan = 0;
        do {
          fgets(line, 128, f); nread += strlen(line);
          if (*line != ';') {
            char opt[3][32]; /* 3 = number of possible optional fields */
            int read, i;
            /* Reference label is optional */
            EEG->eep_header.chanv[chan].reflab[0] = '\0';
            EEG->eep_header.chanv[chan].status[0] = '\0';
            EEG->eep_header.chanv[chan].type[0] = '\0';
            if ((read = sscanf(line, "%10s %lf %lf %10s %32s %32s %32s",
                  EEG->eep_header.chanv[chan].lab,
                  &(EEG->eep_header.chanv[chan].iscale),
                  &(EEG->eep_header.chanv[chan].rscale),
                  EEG->eep_header.chanv[chan].runit,
                  opt[0], opt[1], opt[2])) < 4)
              return 1;
            for (i = 0; i < 3 /* Number of possible optional fields */; i++)
            {
              if (read >= 5 + i) /* Still more arguments? */
              { /* Parse first optional field */
                if (strstr(opt[i], "REF:") == opt[i])
                {
                  strncpy(EEG->eep_header.chanv[chan].reflab, opt[i]+4, 10);
                  EEG->eep_header.chanv[chan].reflab[9] = '\0';
                }
                else if (strstr(opt[i], "STAT:") == opt[i])
                {
                  strncpy(EEG->eep_header.chanv[chan].status, opt[i]+5, 10);
                  EEG->eep_header.chanv[chan].status[9] = '\0';
                }
                else if (strstr(opt[i], "TYPE:") == opt[i])
                {
                  strncpy(EEG->eep_header.chanv[chan].type, opt[i]+5, 10);
                  EEG->eep_header.chanv[chan].type[9] = '\0';
                }
                else if (read == 5) /* No (valid) label but exactly 5 columns enables    */
                {                   /*   workaround for old files: it must be a reflabel */
                  strncpy(EEG->eep_header.chanv[chan].reflab, opt[0], 10);
                  EEG->eep_header.chanv[chan].reflab[9] = '\0';
                }
              }
            }
            chan++;
          }
        } while (chan < EEG->eep_header.chanc);
      }
      else if (strstr(line, "[History]")) {
        eep_set_history(EEG, "");
        do {
          fgets(histline, 2048, f); nread += strlen(histline);
          if (strstr(histline, "EOH") != histline)
            varstr_append(EEG->history, histline);
        } while (strstr(histline, "EOH") != histline);
      }
      /* Averaging (AVR) headers */
      else if (strstr(line, "[Number of averaged Triggers]") || strstr(line, "[Averaged Trials]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%ld", &EEG->eep_header.averaged_trials);
      }
      else if (strstr(line, "[Total Number of Triggers]") || strstr(line, "[Total Trials]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%ld", &EEG->eep_header.total_trials);
      }
      else if (strstr(line, "[Condition Label]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%24s",EEG->eep_header.conditionlabel);
      }
      else if (strstr(line, "[Condition Color]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%24s",EEG->eep_header.conditioncolor);
      }
      else if (strstr(line, "[Pre-stimulus]")) {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%lf",&EEG->eep_header.pre_stimulus);
      }
    }
    // infinite loop protection
    if(nread==nread_last) {
      break;
    }
    nread_last=nread;
  } while (nread < nread_total); // EEG->eeph.size);

  return ferror(f);
}

int writehead_RAW3(eeg_t *EEG, var_string buf)
{
  int chan;

  char line[100];

  sprintf(line, "[File Version]\n%d.%d\n", CNTVERSION_MAJOR, CNTVERSION_MINOR);
  varstr_set(buf, line);
  sprintf(line, "[Sampling Rate]\n%.10f\n", (1000.0 / EEG->eep_header.period) * 0.001 );
  varstr_append(buf, line);
  sprintf(line, "[Samples]\n%" PRIu64 "\n", EEG->eep_header.samplec);
  varstr_append(buf, line);
  sprintf(line, "[Channels]\n%d\n", EEG->eep_header.chanc);
  varstr_append(buf, line);

  sprintf(line, "[Basic Channel Data]\n");
  varstr_append(buf, line);
  sprintf(line, ";label    calibration factor\n");
  varstr_append(buf, line);
  for (chan = 0; chan < EEG->eep_header.chanc; chan++) {
    sprintf(line, "%-10s %.10le %.10le %s",
      EEG->eep_header.chanv[chan].lab,
      EEG->eep_header.chanv[chan].iscale,
      EEG->eep_header.chanv[chan].rscale,
      EEG->eep_header.chanv[chan].runit
    );
    varstr_append(buf, line);
    if (EEG->eep_header.chanv[chan].reflab[0] != '\0')
    {
      sprintf(line, " REF:%-10s", EEG->eep_header.chanv[chan].reflab);
      varstr_append(buf, line);
    }
    if (EEG->eep_header.chanv[chan].status[0] != '\0')
    {
      sprintf(line, " STAT:%-10s", EEG->eep_header.chanv[chan].status);
      varstr_append(buf, line);
    }
    if (EEG->eep_header.chanv[chan].type[0] != '\0')
    {
      sprintf(line, " TYPE:%-10s", EEG->eep_header.chanv[chan].type);
      varstr_append(buf, line);
    }
    varstr_append(buf, "\n");
  }
  if (EEG->eep_header.averaged_trials)
  {
    sprintf(line, "[Averaged Trials]\n%ld\n", EEG->eep_header.averaged_trials);
    varstr_append(buf, line);
  }

  if (EEG->eep_header.total_trials)
  {
    sprintf(line, "[Total Trials]\n%ld\n", EEG->eep_header.total_trials);
    varstr_append(buf, line);
  }

  if (EEG->eep_header.conditionlabel[0] != '\0')
  {
    sprintf(line, "[Condition Label]\n%s\n", EEG->eep_header.conditionlabel);
    varstr_append(buf, line);
  }

  if (EEG->eep_header.conditioncolor[0] != '\0')
  {
    sprintf(line, "[Condition Color]\n%s\n", EEG->eep_header.conditioncolor);
    varstr_append(buf, line);
  }

  if (fabs(EEG->eep_header.pre_stimulus) > 0.00001)
  {
    sprintf(line, "[Pre-stimulus]\n%.10lf\n", EEG->eep_header.pre_stimulus);
    varstr_append(buf, line);
  }

  if (varstr_length(EEG->history) > 0) {
    varstr_append(buf, "[History]\n");
    varstr_append(buf, varstr_cstr(EEG->history));
    varstr_append(buf, "\nEOH\n");
  }

  return CNTERR_NONE;
}

int getepoch_impl(eeg_t *cnt, eep_datatype_e type, uint64_t epoch)
{
  uint64_t insize, insamples, got, samples_to_read;
  char *inbuf;
  storage_t *store = &cnt->store[type];

  uint64_t totsamples = 0;
  switch (type)
  {
    case DATATYPE_EEG:
    case DATATYPE_AVERAGE:
    case DATATYPE_STDDEV:
      totsamples = cnt->eep_header.samplec;
      break;

    case DATATYPE_TIMEFREQ:
      totsamples = cnt->tf_header.samplec;
      break;

    default:
      return CNTERR_BADREQ;
  }

  /* how many bytes to read ? */
  if (epoch == store->epochs.epochc - 1) {
    /* guard for the (unsigned) subtraction below */
	if(totsamples < epoch * store->epochs.epochl)
		return CNTERR_BADREQ;

    insize = store->ch_data.size - store->epochs.epochv[epoch];
    samples_to_read = totsamples - epoch * store->epochs.epochl;
	insamples = (samples_to_read < store->epochs.epochl) ? samples_to_read : store->epochs.epochl;
  }
  else {
    insize = store->epochs.epochv[epoch + 1] - store->epochs.epochv[epoch];
    insamples = store->epochs.epochl;
  }

#ifdef CNT_MMAP
  inbuf = store->data_map + store->map_offset + store->epochs.epochv[epoch];
#else
  /* seek/read source file */
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_seek(cnt->f, store->epochs.epochv[epoch], SEEK_SET, store->ch_data), CNTERR_FILE);
    RET_ON_RIFFERROR(riff_read(store->data.cbuf, sizeof(char), insize, cnt->f, store->ch_data), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_seek(cnt->f, store->epochs.epochv[epoch], SEEK_SET, store->ch_data), CNTERR_FILE);
    RET_ON_RIFFERROR(riff64_read(store->data.cbuf, sizeof(char), insize, cnt->f, store->ch_data), CNTERR_FILE);
  }

  inbuf=store->data.cbuf;
#endif
  store->data.bufepoch = epoch;
  store->data.readpos = 0;

  switch (type)
  {
    case DATATYPE_EEG: /* Read 'normal' EEG data, using RAW3 compression, INT format */
      got = decompepoch_mux(cnt->r3, inbuf, insamples, store->data.buf_int);
      if (got != insize)
      {
        NOT_IN_WINDOWS(fprintf(stderr, "cnt: checksum error: got %" PRIu64 " expected %" PRIu64 " filepos %" PRIu64 " epoch %" PRIu64 "\n", got, insize, store->epochs.epochv[epoch], epoch));
        return CNTERR_DATA;
      }
      break;

    case DATATYPE_TIMEFREQ: /* Read TF data, FLOAT format */
      RET_ON_CNTERROR(tf_convert_for_read(cnt, inbuf, store->data.buf_float, insamples));
      break;

    case DATATYPE_AVERAGE: /* Read AVR data, FLOAT format */
    case DATATYPE_STDDEV:  /* Read Standard deviation data, FLOAT format */
      RET_ON_CNTERROR(rawf_convert_for_read(cnt, inbuf, store->data.buf_float, insamples));
      break;

    default:
      return CNTERR_DATA;
      break;
  }
  return CNTERR_NONE;
}

int putepoch_impl(eeg_t *cnt)
{
  // int smp = 0;
  uint64_t to_write;
  storage_t *store;

  if ((DATATYPE_UNDEFINED == cnt->current_datachunk) ||
      (!cnt->store[cnt->current_datachunk].initialized))
      return CNTERR_BADREQ;

  store = &cnt->store[cnt->current_datachunk];

  /* unwritten data in buffers ? - prepare for writing */
  if (store->data.writepos != 0)
  {
    switch (cnt->current_datachunk)
    {
      case DATATYPE_EEG:
        to_write = compepoch_mux(cnt->r3, store->data.buf_int, (int) store->data.writepos, store->data.cbuf);
        break;

      case DATATYPE_AVERAGE:
      case DATATYPE_STDDEV:
        to_write = rawf_convert_for_write(cnt, store->data.buf_float, store->data.cbuf, store->data.writepos);
        break;

      case DATATYPE_TIMEFREQ:
        to_write = tf_convert_for_write(cnt, store->data.buf_float, store->data.cbuf, store->data.writepos);
        break;

      default:
        return CNTERR_BADREQ; /* Invalid mode, this should not happen */
    }

    /* write the filled buffers to file, reset buffers */
    if(cnt->mode==CNT_RIFF) {
      RET_ON_RIFFERROR(riff_write(store->data.cbuf, sizeof(char), to_write, cnt->f, &store->ch_data), CNTERR_FILE);
    } else {
      RET_ON_RIFFERROR(riff64_write(store->data.cbuf, sizeof(char), to_write, cnt->f, &store->ch_data), CNTERR_FILE);
    }

    if (DATATYPE_TIMEFREQ == cnt->current_datachunk )
      cnt->tf_header.samplec += store->data.writepos;
    else
      cnt->eep_header.samplec += store->data.writepos;

    store->data.writepos = 0;

    /* register access info for this buffer */
    store->epochs.epochv = (uint64_t *) v_realloc(store->epochs.epochv, (size_t) (store->epochs.epochc + 1) * sizeof(uint64_t), "epv");
    store->epochs.epochv[store->epochs.epochc] = store->epochs.epvbuf;
    store->epochs.epochc++;

    /* prepare registering next buffer */
    store->epochs.epvbuf += to_write;
  }
  if (cnt->keep_consistent)
    make_partial_output_consistent(cnt, 0 /* No finalize */);
  return CNTERR_NONE;
}

int gethead_TFH(eeg_t *EEG, chunk_t* chunk, tf_header_t* tf_header)
{
  FILE *f = EEG->f;
  int nread = 0;
  char line[128], line2[128], *descr, *descr_end;
  int comp;
  double rate = -1.0;

  do
  {
    fgets(line, 128, f); nread += strlen(line);

    if (*line == '[') {
    if (strstr(line, "[Sampling Rate]"))
    {
      fgets(line, 128, f); nread += strlen(line);
    if (sscanf(line, "%lf", &rate) != 1 || rate < 1e-30)
      return 1;
        tf_header->period = 1.0 / rate;
    }
    else if (strstr(line, "[Samples]"))
    {
      fgets(line, 128, f); nread += strlen(line);
    if (sscanf(line, "%" SCNu64, &tf_header->samplec) != 1)
      return 1;
    }
      else if (strstr(line, "[TimeFrequencyType]"))
    {
        fgets(line, 128, f); nread += strlen(line);
        if (sscanf(line, "%40s", tf_header->tf_type) != 1)
          return 1;
      }
      else if (strstr(line, "[ContentDataType]"))
    {
        fgets(line, 128, f); nread += strlen(line);

    if (sscanf(line, "%s", line2) != 1)
    {
      tf_header->content_datatype = CONTENT_UNKNOWN;
      return 1;
    }
    if (!strcmp(line2, "REAL"))
      tf_header->content_datatype = COMPLEX_REAL;
    else if (!strcmp(line2, "IMAGINARY"))
      tf_header->content_datatype = COMPLEX_IMAGINARY;
    else if (!strcmp(line2, "PHASE"))
      tf_header->content_datatype = COMPLEX_PHASE;
    else if (!strcmp(line2, "ABSOLUTE"))
      tf_header->content_datatype = COMPLEX_ABSOLUTE;
    else if (!strcmp(line2, "POWER"))
      tf_header->content_datatype = CONTENT_POWER;
    else
    {
      tf_header->content_datatype = CONTENT_UNKNOWN;
      return 1;
    }
      }
      else if (strstr(line, "[Components]"))
    {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%" PRIu64 "", &tf_header->componentc);
      }
    else if (strstr(line, "[Unit]"))
    {
        fgets(line, 128, f); nread += strlen(line);
        sscanf(line, "%15s", tf_header->tf_unit);
      }
      else if (strstr(line, "[Basic Component Data]"))
    {
        if (tf_header->componentc < 1)
          return 1;
        tf_header->componentv = (tf_component_t *)
          v_malloc(tf_header->componentc * sizeof(tf_component_t), "componentv");
        comp = 0;
        do
    {
          fgets(line, 128, f); nread += strlen(line);
          if (*line != ';')
      {
      strcpy(tf_header->componentv[comp].description, "");
            if (sscanf(line, "%f", &tf_header->componentv[comp].axis_value) != 1)
        return 1;
      /* The description may contain spaces, so it should always be between double
        quotes. Unfortunately, we can't use scanf to parse this correctly */
      descr = strchr(line, '"');
      if (NULL != descr)
      {
        descr++;
        descr_end = strchr(descr, '"');
        if (NULL == descr_end)
          return 1;
        else
          *descr_end = '\0';
        strcpy(tf_header->componentv[comp].description, descr);
      }
      comp++;
          }
        } while (comp < tf_header->componentc);
      }
    }
  } while (nread < chunk->size);

  return ferror(f);
}

/* Converts the Time/frequency header into an ASCII string */
int writehead_TFH(tf_header_t* tf_header, var_string buf)
{
  uint64_t comp;
  char line[100];

  sprintf(line, "[TimeFrequencyType]\n%s\n", tf_header->tf_type);

  varstr_set(buf, line);

  strcpy(line, "[ContentDataType]\n");

  switch (tf_header->content_datatype)
  {
  case COMPLEX_REAL:
    strcat(line, "REAL\n");
    break;
  case COMPLEX_IMAGINARY:
    strcat(line, "IMAGINARY\n");
    break;
  case COMPLEX_PHASE:
    strcat(line, "PHASE\n");
    break;
  case COMPLEX_ABSOLUTE:
    strcat(line, "ABSOLUTE\n");
    break;
  case CONTENT_POWER:
    strcat(line, "POWER\n");
    break;

  default: /* This is an error! */
    return CNTERR_DATA;
  }
  varstr_append(buf, line);

  sprintf(line, "[Sampling Rate]\n%-4.3f\n", 1.0 / tf_header->period);
  varstr_append(buf, line);

  sprintf(line, "[Samples]\n%" PRIu64 "\n", tf_header->samplec);
  varstr_append(buf, line);

  sprintf(line, "[Components]\n%" PRIu64 "\n", tf_header->componentc);
  varstr_append(buf, line);

  varstr_append(buf, "[Basic Component Data]\n");
  varstr_append(buf, "; component axis value, component description (opt)\n");
  for (comp = 0; comp < tf_header->componentc; comp++)
  {
    sprintf(line, "%-4.6f", tf_header->componentv[comp].axis_value);
    varstr_append(buf, line);
    if (strlen(tf_header->componentv[comp].description) > 0)
    {
      sprintf(line, " \"%s\"", tf_header->componentv[comp].description);
      varstr_append(buf, line);
    }
    varstr_append(buf, "\n");
  }

  sprintf(line, "[Unit]\n%s\n", tf_header->tf_unit);
  varstr_append(buf, line);

  return CNTERR_NONE;
}

int tf_convert_for_read(eeg_t *cnt, char *in, float* out, int length)
{
  char *inbase;
  float *outbase;
  uint64_t comp, chan, smpstep, smp, seq_id;

  smpstep = cnt->tf_header.componentc * cnt->eep_header.chanc;
  for (seq_id = 0; seq_id < cnt->tf_header.componentc * cnt->eep_header.chanc; seq_id++)
  {
    comp = cnt->store[DATATYPE_TIMEFREQ].chanseq[2*seq_id];
    chan = cnt->store[DATATYPE_TIMEFREQ].chanseq[2*seq_id+1];
    inbase = &in[seq_id * (length*sizeof(float)+1)];
    outbase = &out[comp * cnt->eep_header.chanc + chan];
    if (UNCOMPR_FLOAT_32 == inbase[0])
    {
      for (smp = 0; smp < length; smp++)
        sread_f32(&inbase[sizeof(float)*smp+1], &outbase[smp * smpstep]);
    }
    else
      return CNTERR_DATA;
  }
  return CNTERR_NONE;
}

long tf_convert_for_write(eeg_t *cnt, float *in, char* out, int length)
{
  float *inbase;
  char *outbase;
  uint64_t comp, chan, smpstep, smp, seq_id;

  smpstep = cnt->tf_header.componentc * cnt->eep_header.chanc;
  for (seq_id = 0; seq_id < cnt->tf_header.componentc * cnt->eep_header.chanc; seq_id++)
  {
    comp = cnt->store[DATATYPE_TIMEFREQ].chanseq[2*seq_id];
    chan = cnt->store[DATATYPE_TIMEFREQ].chanseq[2*seq_id+1];
    inbase = &in[comp * cnt->eep_header.chanc + chan];
    outbase = &out[seq_id * (sizeof(float)*length + 1)];
    outbase[0] = UNCOMPR_FLOAT_32;
    for (smp = 0; smp < length; smp++)
      swrite_f32(&outbase[sizeof(float)*smp+1], inbase[smp * smpstep]);
  }
  return TF_CBUF_SIZE(cnt, length);
}

int rawf_convert_for_read(eeg_t *cnt, char *in, float* out, int length)
{
  char *inbase;
  float *outbase;
  uint64_t smpstep, smp, seq_id;

  smpstep = cnt->eep_header.chanc;
  for (seq_id = 0; seq_id < cnt->eep_header.chanc; seq_id++)
  {
    inbase = &in[seq_id * (length*sizeof(float)+1)];
    outbase = &out[cnt->store[DATATYPE_AVERAGE].chanseq[seq_id]];
    if (UNCOMPR_FLOAT_32 == inbase[0])
    {
      for (smp = 0; smp < length; smp++)
        sread_f32(&inbase[sizeof(float)*smp+1], &outbase[smp * smpstep]);
    }
    else
      return CNTERR_DATA;
  }
  return CNTERR_NONE;
}

long rawf_convert_for_write(eeg_t *cnt, float *in, char* out, int length)
{
  float *inbase;
  char *outbase;
  uint64_t smpstep, smp, seq_id;

  smpstep = cnt->eep_header.chanc;
  for (seq_id = 0; seq_id < cnt->eep_header.chanc; seq_id++)
  {
    inbase = &in[cnt->store[DATATYPE_AVERAGE].chanseq[seq_id]];
    outbase = &out[seq_id * (sizeof(float)*length + 1)];
    outbase[0] = UNCOMPR_FLOAT_32;
    for (smp = 0; smp < length; smp++)
      swrite_f32(&outbase[sizeof(float)*smp+1], inbase[smp * smpstep]);
  }
  return FLOAT_CBUF_SIZE(cnt, length);
}

/* general constructor/destructor ------------------------------- */
eeg_t *cnt_init()
{
  eeg_t *cnt = (eeg_t *) v_malloc(sizeof(eeg_t), "cnt");
  memset(cnt, 0, sizeof(eeg_t));
  /* Initialize some non-zero values */
  cnt->store[DATATYPE_EEG].fourcc = FOURCC_raw3;
  cnt->store[DATATYPE_EEG].data.bufepoch = -2;     /* tricky - see getepoch_RAW3 */
  cnt->store[DATATYPE_TIMEFREQ].fourcc = FOURCC_tfd;
  cnt->store[DATATYPE_TIMEFREQ].data.bufepoch = -2;
  cnt->store[DATATYPE_AVERAGE].fourcc = FOURCC_rawf;
  cnt->store[DATATYPE_AVERAGE].data.bufepoch = -2;
  cnt->store[DATATYPE_STDDEV].fourcc = FOURCC_stdd;
  cnt->store[DATATYPE_STDDEV].data.bufepoch = -2;
  //cnt->active_chunk_mode = CHUNKMODE_UNDEFINED;
  cnt->current_datachunk = DATATYPE_UNDEFINED;
  cnt->tf_header.content_datatype = CONTENT_UNKNOWN;
  cnt->f = 0;

  return cnt;
}

void storage_free(storage_t *store)
{
  v_free(store->epochs.epochv);
  v_free(store->chanseq);
  v_free(store->data.buf_int);
  v_free(store->data.buf_float);
  v_free(store->data.cbuf);
#ifdef CNT_MMAP
  if (store->data_mapped)
  {
    if (munmap(store->data_map, store->ch_data.size))
      NOT_IN_WINDOWS(fprintf(stderr, "cnt: munmap() failed\n"));
  }
#endif

}

void eep_free(eeg_t *cnt)
{
  raw3_free(cnt->r3);

  /* trigger chunk: free list */
  trg_free(cnt->trg);

  storage_free(&cnt->store[DATATYPE_EEG]);
  storage_free(&cnt->store[DATATYPE_AVERAGE]);
  storage_free(&cnt->store[DATATYPE_TIMEFREQ]);
  storage_free(&cnt->store[DATATYPE_STDDEV]);

  v_free(cnt->eep_header.chanv);
  v_free(cnt->tf_header.componentv);

  /* Free recording info */
  v_free(cnt->recording_info);

  v_free(cnt->fname);

  /* Free history */
  varstr_destruct(cnt->history);

  /* Free values */
  val_destroy(cnt->values);

  /* Finally, free the eeg_t structure itself */
  v_free(cnt);
}

eeg_t *eep_init_from_values(double period, short chanc, eegchan_t *chanv)
{
  eeg_t *cnt;

  if(chanc < 1 || chanc > CNT_MAX_CHANC)
    return NULL;

  cnt = cnt_init();

  cnt->eep_header.period = period;
  cnt->eep_header.chanc = chanc;
  cnt->eep_header.chanv = chanv;
  cnt->trg = trg_init();   /* empty trigger table */

  return cnt;
}

eeg_t* eep_init_from_tf_values(double period, short chanc, eegchan_t *chanv, short compc, tf_component_t *compv)
{
  eeg_t *cnt;

  if(chanc < 1 || chanc > CNT_MAX_CHANC)
    return NULL;

  cnt = eep_init_from_values(period, chanc, chanv);
  cnt->tf_header.componentc = compc;
  cnt->tf_header.componentv = compv;
  cnt->tf_header.period = period;
  return cnt;
}

/* open existing cnt's ------------------------------------------ */
int cntopen_EEP20(eeg_t *EEG);
int cntopen_RAW3(eeg_t *EEG);
int cntopen_NS30(eeg_t *EEG);
int cntopen_AVR(eeg_t *EEG);

eeg_t *eep_init_from_file(const char *fname, FILE *f, int *status)
{
  eeg_t *cnt = cnt_init();
  char filetag[32];
  char avrtag[4] = { 0x26, 0x00, 0x10, 0x00 }; /* 38 16 avr-header-sizes */

  /* register file info */
  cnt->f = f;
  cnt->fname = v_strnew(fname, 0);

  /* read first "magic" bytes to determine filetype */
  if (eepio_fseek(f, 0, SEEK_SET) || eepio_fread(filetag, 16, 1, f) < 1 || eepio_fseek(f, 0, SEEK_SET)) {
    *status = CNTERR_FILE;
  }
  else {
    if ( (!strncmp("RIFF", filetag, 4) && !strncmp("CNT ", &filetag[8], 4)) ||
         (!strncmp("RF64", filetag, 4) && !strncmp("CNT ", &filetag[12], 4)) )
      *status = cntopen_RAW3(cnt);
    else if (!strncmp(TAG_EEP20, filetag, strlen(TAG_EEP20)))
      *status = cntopen_EEP20(cnt);
    else if (!strncmp(TAG_NS30, filetag, strlen(TAG_NS30)))
      *status = cntopen_NS30(cnt);
    else if (!memcmp(avrtag, filetag, 4))
      *status = cntopen_AVR(cnt);
    else
      *status = CNTERR_DATA;
  }

  if (*status != CNTERR_NONE) {
    eep_free(cnt);
    cnt = NULL;
  }

  return cnt;
}

/* Private helper functions for opening/reading RAW3/TF data */

int read_trigger_chunk(eeg_t *EEG)
{
  int riff_open_result;
  if(EEG->mode==CNT_RIFF) {
    riff_open_result = riff_open(EEG->f, &EEG->evt, FOURCC_evt, EEG->cnt);
  } else {
    riff_open_result = riff64_open(EEG->f, &EEG->evt, FOURCC_evt, EEG->cnt);
  }

  if(riff_open_result) {
    /* only issue warning when in debug mode - and when not in windows */
    NOT_IN_WINDOWS(eepdebug("cnt: warning - no <evt > chunk found in %s!\n", EEG->fname));
    EEG->trg = trg_init();
  } else {
    uint64_t i, j;

    EEG->trg = trg_init();
    if(EEG->mode==CNT_RIFF) {
      EEG->trg->c = riff_get_chunk_size(EEG->evt) / 12;
    } else {
      EEG->trg->c = riff64_get_chunk_size(EEG->evt) / 16;
    }
    EEG->trg->v = (trgentry_t *) v_malloc((size_t) EEG->trg->c * sizeof(trgentry_t), "trgv");
    EEG->trg->cmax = EEG->trg->c;

    for (i = 0, j = 0; i < EEG->trg->c; i++) {
      if(EEG->mode==CNT_RIFF) {
        int32_t itmp;
        read_s32(EEG->f, &itmp);
        (EEG->trg->v)[j].sample = itmp;
      } else {
        read_u64(EEG->f, &((EEG->trg->v)[j].sample));
      }
      eepio_fread((EEG->trg->v)[j].code, TRG_CODE_LENGTH, 1, EEG->f);
      (EEG->trg->v)[j].code[TRG_CODE_LENGTH] = '\0';

      /*
      a buggy early version wrote each DISCONT and DCRESET twice
      (both the code and the EEP 2.0 flag value)
      we simply ignore the values here
      it's safe to remove this code if no more 1996 data are present...
      */
      if (    j > 0
          && (EEG->trg->v)[j - 1].sample == (EEG->trg->v)[j].sample
          && (    !strcmp((EEG->trg->v)[j].code, "1024")
          || !strcmp((EEG->trg->v)[j].code, "2048"))   )
      {
      }
      else {
        j++;
      }
    }
    EEG->trg->c = j;
  }
  return CNTERR_NONE;
}

int read_epoch_chunk(eeg_t *EEG, storage_t *store)
{
  int32_t itmp;
  uint64_t epoch;

  if(EEG->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_list_open(EEG->f, &store->ch_toplevel, store->fourcc, EEG->cnt), CNTERR_DATA);
    RET_ON_RIFFERROR(riff_open(EEG->f, &store->ch_ep, FOURCC_ep, store->ch_toplevel), CNTERR_DATA);

    read_s32(EEG->f, &itmp);
    store->epochs.epochl = itmp;

    store->epochs.epochc = store->ch_ep.size / 4 - 1;
  } else {
    RET_ON_RIFFERROR(riff64_list_open(EEG->f, &store->ch_toplevel, store->fourcc, EEG->cnt), CNTERR_DATA);
    RET_ON_RIFFERROR(riff64_open(EEG->f, &store->ch_ep, FOURCC_ep, store->ch_toplevel), CNTERR_DATA);

    read_u64(EEG->f, &store->epochs.epochl);
    store->epochs.epochc = store->ch_ep.size / 8 - 1;
  }

  if (store->epochs.epochc <= 0 || store->epochs.epochl <= 0)
    return CNTERR_DATA;

  store->epochs.epochv = (uint64_t *) v_malloc((size_t) store->epochs.epochc * sizeof(uint64_t), "epochv");
  for (epoch = 0; epoch < store->epochs.epochc; epoch++) {
    if(EEG->mode==CNT_RIFF) {
      int32_t itmp;
      read_s32(EEG->f, &itmp);
      store->epochs.epochv[epoch] = (uint64_t)itmp;
    } else {
      read_u64(EEG->f, &store->epochs.epochv[epoch]);
    }
  }

  return CNTERR_NONE;
}

#ifdef CNT_MMAP
int mmap_data_chunk(FILE *f, storage_t *store)
{
  /*EEG->data_map = mmap(0, EEG->rawdata.ch_data.start + 8 + EEG->rawdata.ch_data.size,
                       PROT_READ, MAP_FILE | MAP_PRIVATE, fileno(EEG->f), 0);*/
  int offset = ((store->ch_data.start + 8) / getpagesize()) * getpagesize();
  store->map_offset = (store->ch_data.start + 8) - offset;
  store->data_map =
   mmap(0, store->ch_data.size + store->map_offset, PROT_READ, MAP_FILE | MAP_SHARED,
        fileno(f), offset);
  if (store->data_map == MAP_FAILED)
  {
    NOT_IN_WINDOWS(fprintf(stderr, "cnt: mmap() failed\n"));
    return CNTERR_FILE;
  }
  store->data_mapped = 1;
  return CNTERR_NONE;
}
#endif

int read_chanseq_chunk(eeg_t *EEG, storage_t *store, uint64_t expected_length)
{
  int chanin;
  uint64_t compchan;

  if(EEG->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_list_open(EEG->f, &store->ch_toplevel, store->fourcc, EEG->cnt), CNTERR_DATA);
    RET_ON_RIFFERROR(riff_open(EEG->f, &store->ch_chan, FOURCC_chan, store->ch_toplevel), CNTERR_DATA);
  } else {
    RET_ON_RIFFERROR(riff64_list_open(EEG->f, &store->ch_toplevel, store->fourcc, EEG->cnt), CNTERR_DATA);
    RET_ON_RIFFERROR(riff64_open(EEG->f, &store->ch_chan, FOURCC_chan, store->ch_toplevel), CNTERR_DATA);
  }
  if (store->ch_chan.size != expected_length * sizeof(short))
    return CNTERR_DATA;
  store->chanseq = (short *) v_malloc(expected_length * sizeof(short), "tf_chanseq");
  for (compchan = 0; compchan < expected_length; compchan++)
  {
    read_s16(EEG->f, &chanin);
    store->chanseq[compchan] = (short) chanin;
  }
  return CNTERR_NONE;

}

int _cntopen_raw3(eeg_t *EEG) {
  int i;
  FILE *f = EEG->f;
  fourcc_t formtype;
  chunk_t dummychunk;

  /* Open the CNT file, and check if it really is a valid CNT file */
  RET_ON_RIFFERROR(riff_form_open(f, &EEG->cnt, &formtype), CNTERR_DATA);
  RET_ON_RIFFERROR(riff_open(f, &dummychunk, FOURCC_eeph, EEG->cnt), CNTERR_DATA);

  EEG->mode = CNT_RIFF;

  /* There should always be an EEP header */
  RET_ON_RIFFERROR(riff_open(f, &EEG->eeph, FOURCC_eeph, EEG->cnt), CNTERR_DATA);
  if (gethead_RAW3(EEG)) {
    return CNTERR_FILE;
  }

  if (RIFFERR_NONE == riff_list_open(f, &dummychunk, FOURCC_tfd, EEG->cnt))
  { /* The file contains Time/Freq data, so there MUST be a Time Freq. Header */
    RET_ON_RIFFERROR(riff_open(f, &EEG->tfh, FOURCC_tfh, EEG->cnt), CNTERR_DATA);
    if (gethead_TFH(EEG, &EEG->tfh, &EEG->tf_header))
      return CNTERR_FILE;
  }

  /* Check major file version (if it is larger then the library version,
    we can't read this file) */
  if (EEG->eep_header.fileversion_major > CNTVERSION_MAJOR)
    return CNTERR_DATA;

  /* read trigger table (event list) */
  RET_ON_CNTERROR(read_trigger_chunk(EEG));

  /* Read recording info - if it's there */
  EEG->recording_info = (record_info_t*) malloc (sizeof(record_info_t));
  if (NULL != EEG->recording_info)
      if (CNTERR_NONE != read_recinfo_chunk(EEG, EEG->recording_info))
        v_free(EEG->recording_info);

  for (i = 0; i < NUM_DATATYPES; i++)
    init_data_store(EEG, (eep_datatype_e) i);

  return CNTERR_NONE;
}

int _cntopen_raw3_64(eeg_t *EEG) {
  int i;
  FILE *f = EEG->f;
  fourcc_t formtype;
  chunk64_t dummychunk;

  /* Open the CNT file, and check if it really is a valid CNT file */
  RET_ON_RIFFERROR(riff64_form_open(f, &EEG->cnt, &formtype), CNTERR_DATA);
  RET_ON_RIFFERROR(riff64_open(f, &dummychunk, FOURCC_eeph, EEG->cnt), CNTERR_DATA);

  EEG->mode = CNTX_RIFF;

  /* There should always be an EEP header */
  RET_ON_RIFFERROR(riff64_open(f, &EEG->eeph, FOURCC_eeph, EEG->cnt), CNTERR_DATA);
  if (gethead_RAW3(EEG)) {
    return CNTERR_FILE;
  }
  // The file contains Time/Freq data, so there MUST be a Time Freq. Header */
  if (RIFFERR_NONE == riff64_list_open(f, &dummychunk, FOURCC_tfd, EEG->cnt)) {
    // TODO, unsupported time freq data
    RET_ON_RIFFERROR(riff64_open(f, &EEG->tfh, FOURCC_tfh, EEG->cnt), CNTERR_DATA);
    if (gethead_TFH(EEG, &EEG->tfh, &EEG->tf_header))
      return CNTERR_FILE;
  }

  /* Check major file version (if it is larger then the library version,
    we can't read this file) */
  if (EEG->eep_header.fileversion_major > CNTVERSION_MAJOR)
    return CNTERR_DATA;

  /* read trigger table (event list) */
  RET_ON_CNTERROR(read_trigger_chunk(EEG));

  /* Read recording info - if it's there */
  EEG->recording_info = (record_info_t*) malloc (sizeof(record_info_t));
  if (NULL != EEG->recording_info)
      if (CNTERR_NONE != read_recinfo_chunk(EEG, EEG->recording_info))
        v_free(EEG->recording_info);

  for (i = 0; i < NUM_DATATYPES; i++)
    init_data_store(EEG, (eep_datatype_e) i);

  return CNTERR_NONE;
}

int cntopen_RAW3(eeg_t *EEG) {
  fourcc_t formtype;

  eepio_fseek(EEG->f, 0, SEEK_SET);
  eepio_fread(&formtype, 4, 1, EEG->f);
  if(formtype==FOURCC_RIFF) {
    return _cntopen_raw3(EEG);
  }

  if(formtype==FOURCC_RF64) {
    return _cntopen_raw3_64(EEG);
  }

  return CNTERR_DATA;
}

/* derive a new eeg structure from opened source ------------------------- */

eeg_t *eep_init_from_copy(eeg_t *src)
{
  /* clean startup */
  eeg_t *dst = cnt_init();

  /* copy basic values from src to new */
  dst->mode = src->mode;

  memcpy(&dst->eep_header, &src->eep_header, sizeof(eep_header_t));
  dst->eep_header.samplec = 0; /* Reset samplecount */
  dst->eep_header.chunk_size = 0;

  /*dst->eep_header.period = src->eep_header.period;
  dst->eep_header.chanc = src->eep_header.chanc;*/
  dst->eep_header.chanv = (eegchan_t *) v_malloc(dst->eep_header.chanc * sizeof(eegchan_t), "chanv");
  memcpy(dst->eep_header.chanv, src->eep_header.chanv, dst->eep_header.chanc * sizeof(eegchan_t));

  /* copy history table */
  eep_set_history(dst, eep_get_history(src));

  /* copy trigger archive, if present */
  dst->trg = trg_init();
  if (src->trg && src->trg->c) {
    dst->trg->c = src->trg->c;
    dst->trg->v = (trgentry_t *) v_malloc((size_t) dst->trg->c * sizeof(trgentry_t), "trgv");
    dst->trg->cmax = src->trg->c;
    memcpy(dst->trg->v, src->trg->v, (size_t) dst->trg->c * sizeof(trgentry_t));
  }

  /* copy compression stuff */
  if ( (dst->mode == CNT_RIFF || dst->mode == CNTX_RIFF) && src->store[DATATYPE_TIMEFREQ].initialized)
  {
    /*for (i = 0; i < NUM_DATATYPES; i++)
    {
      switch (i)
      {
        case DATATYPE_EEG:
        case DATATYPE_AVERAGE:
        case DATATYPE_STDDEV:
          if (src->store[i].initialized)
          { */
            /*dst->store[i].epochs.epochl = src->store[i].epochs.epochl;
            dst->store[i].chanseq = (short *)
              v_malloc(dst->eep_header.chanc * sizeof(short), "chanseq");
            memcpy(dst->store[i].chanseq, src->store[i].chanseq, dst->eep_header.chanc * sizeof(short));*/
            /*init_data_store(dst, i, 0 ); */
          /*}
          break;

        case DATATYPE_TIMEFREQ:
          if (src->store[i].initialized)
          {*/
            /* Copy time/frequency header */
            memcpy(&dst->tf_header, &src->tf_header, sizeof(tf_header_t));
            dst->tf_header.samplec = 0; /* Reset samplecount, we want to create an 'empty' struct */
            dst->tf_header.chunk_size = 0;
            /* Deepcopy the component list contained in the tf_header */
            dst->tf_header.componentv = (tf_component_t*)
              v_malloc(dst->tf_header.componentc * sizeof(tf_component_t), "tf_header.componentv");
            memcpy(dst->tf_header.componentv, src->tf_header.componentv,
                dst->tf_header.componentc * sizeof(tf_component_t));
            /* Copy the component/channel sequence list */
            /*dst->store[i].chanseq = (short *)
              v_malloc(dst->eep_header.chanc * dst->tf_header.componentc * sizeof(short), "tf_chanseq");
            memcpy(dst->store[i].chanseq, src->store[i].chanseq,
              dst->eep_header.chanc * dst->tf_header.componentc * sizeof(short)); */
            /* Copy epoch length */
            /*dst->store[i].epochs.epochl = src->store[i].epochs.epochl;*/
            /* Do not copy the epochs themselves as we want to start with an 'empty' struct */
            /*init_data_store(dst, i, 0);*/
          /*}
          break;

        default:
          eep_free(dst);
          return NULL;
      }
    }*/
  }

  /* NeuroScan and avr input is silently converted to riff output */
  if (dst->mode == CNT_NS30 || dst->mode == CNT_AVR) {
    dst->mode = CNT_RIFF;
    /*epochl = (int) (1.0 / dst->eep_header.period);
    if (epochl < 100) epochl = 100;
    if (epochl > 1000) epochl = 1000;
    dst->store[DATATYPE_EEG].epochs.epochl = epochl;*/

    /*init_data_store(dst, DATATYPE_EEG);
      eep_prepare_to_write(dst, DATATYPE_EEG, epochl, NULL); */
    /* NB: This functionality is broken. If an existing program really
       depends on it it should be changed to call
       eep_prepare_to_write with a new epochlength, after calling eep_create_file. */
  }

  return dst;
}

/* create a output cnt file according to the initialized(!) dst eeg_t ---- */
int eep_create_file(eeg_t *dst, const char *fname, FILE *f, eeg_t *src, unsigned long delmask, const char *registry) {
  // long forgetmask = 0;

  /* register file info */
  dst->mode = CNT_RIFF;
  dst->f = f;
  dst->fname = (char *) v_malloc(strlen(fname) + 1,"fname");
  strcpy(dst->fname, fname);

  /* extend history */
  eep_append_history(dst, registry);

  /* initiate the dest file riff tree */
  RET_ON_RIFFERROR(riff_form_new(f, &dst->cnt, FOURCC_CNT), CNTERR_FILE);

  /* copy unknown chunks from src to dst */
  if (src != NULL)
    RET_ON_CNTERROR(saveold_RAW3(dst, src, delmask));

  if (CNT_RIFF != dst->mode)
    return eep_create_file_EEP20(dst, src, delmask);

  /*for (i = 0; i < NUM_DATATYPES; i++)
    if (dst->store[i].initialized)
      return eep_prepare_to_write(dst, i);*/

  return CNTERR_NONE;
}

int eep_create_file64(eeg_t *dst, const char *fname, FILE *f, const char *registry) {
  // long forgetmask = 0;

  /* register file info */
  dst->mode = CNTX_RIFF;
  dst->f = f;
  dst->fname = (char *) v_malloc(strlen(fname) + 1,"fname");
  strcpy(dst->fname, fname);

  /* extend history */
  eep_append_history(dst, registry);

  /* initiate the dest file riff tree */
  RET_ON_RIFFERROR(riff64_form_new(f, &dst->cnt, FOURCC_CNT), CNTERR_FILE);

  return CNTERR_NONE;
}

void eep_clear_epochs(eeg_t *cnt, storage_t *store)
{
  store->epochs.epochc = 0;
  store->epochs.epochv = NULL;
  store->epochs.epvbuf = 0;

  store->data.writepos = 0;
  store->data.bufepoch = 0;
}

int cnt_create_raw3_compr_buffer(eeg_t *EEG)
{
  storage_t *store = &EEG->store[DATATYPE_EEG];
  EEG->r3 = raw3_init(EEG->eep_header.chanc, store->chanseq, store->epochs.epochl);
  store->data.buf_int = (sraw_t *)
    v_malloc((size_t) store->epochs.epochl * EEG->eep_header.chanc * sizeof(sraw_t), "buf");
  store->data.cbuf = (char *)
    v_malloc((size_t) RAW3_EPOCH_SIZE(store->epochs.epochl, EEG->eep_header.chanc), "cbuf");
  if (!EEG->r3 || !store->data.buf_int || !store->data.cbuf)
  {
    v_free(store->data.cbuf);
    v_free(store->data.buf_int);
    raw3_free(EEG->r3);
    return CNTERR_MEM;
  }
  return CNTERR_NONE;
}

int write_chanseq_chunk(eeg_t *cnt, storage_t* store, uint64_t num_chans)
{
  uint64_t chan;
  char outchan[2];
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &store->ch_chan, FOURCC_chan, &store->ch_toplevel), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &store->ch_chan, FOURCC_chan, &store->ch_toplevel), CNTERR_FILE);
  }
  for (chan = 0; chan < num_chans; chan++) {
    swrite_s16(outchan, store->chanseq[chan]);
    if(cnt->mode==CNT_RIFF) {
      RET_ON_RIFFERROR(riff_write(outchan, 2, 1, cnt->f, &store->ch_chan), CNTERR_FILE);
    } else {
      RET_ON_RIFFERROR(riff64_write(outchan, 2, 1, cnt->f, &store->ch_chan), CNTERR_FILE);
    }
  }
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_close(cnt->f, store->ch_chan), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_close(cnt->f, store->ch_chan), CNTERR_FILE);
  }
  return CNTERR_NONE;
}

int write_epoch_chunk(eeg_t *cnt, storage_t *store)
{
  char outepoch[8];
  uint64_t epoch;

  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &store->ch_ep, FOURCC_ep, &store->ch_toplevel), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &store->ch_ep, FOURCC_ep, &store->ch_toplevel), CNTERR_FILE);
  }
  if(cnt->mode==CNT_RIFF) {
    swrite_s32(outepoch, store->epochs.epochl);
    RET_ON_RIFFERROR(riff_write(outepoch, 4, 1, cnt->f, &store->ch_ep), CNTERR_FILE);
  } else {
    swrite_u64(outepoch, store->epochs.epochl);
    RET_ON_RIFFERROR(riff64_write(outepoch, 8, 1, cnt->f, &store->ch_ep), CNTERR_FILE);
  }
  for (epoch = 0; epoch < store->epochs.epochc; epoch++) {
    if(cnt->mode==CNT_RIFF) {
      swrite_s32(outepoch, store->epochs.epochv[epoch]);
      RET_ON_RIFFERROR(riff_write(outepoch, 4, 1, cnt->f, &store->ch_ep), CNTERR_FILE);
    } else {
      swrite_u64(outepoch, store->epochs.epochv[epoch]);
      RET_ON_RIFFERROR(riff64_write(outepoch, 8, 1, cnt->f, &store->ch_ep), CNTERR_FILE);
    }
  }
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_close(cnt->f, store->ch_ep), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_close(cnt->f, store->ch_ep), CNTERR_FILE);
  }
  return CNTERR_NONE;
}

int write_eeph_chunk(eeg_t *cnt)
{
  var_string textbuf;
  int retcode;
  /* Create the EEPH chunk */
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &cnt->eeph, FOURCC_eeph, &cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &cnt->eeph, FOURCC_eeph, &cnt->cnt), CNTERR_FILE);
  }
  textbuf = varstr_construct(); /* Create header string */
  if (NULL == textbuf)
    return CNTERR_MEM;
  writehead_RAW3(cnt, textbuf); /* Fill header string */
  if(cnt->mode==CNT_RIFF) {
    retcode = riff_write(varstr_cstr(textbuf), varstr_length(textbuf), 1, cnt->f, &cnt->eeph); /* Write header to chunk */
  } else {
    retcode = riff64_write(varstr_cstr(textbuf), varstr_length(textbuf), 1, cnt->f, &cnt->eeph); /* Write header to chunk */
  }
  varstr_destruct(textbuf); /* Destroy header string */
  if (RIFFERR_NONE != retcode) /* Return on writing error */
  return CNTERR_FILE;

  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_close(cnt->f, cnt->eeph), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_close(cnt->f, cnt->eeph), CNTERR_FILE);
  }
  return CNTERR_NONE;
}

int write_trigger_chunk(eeg_t *cnt)
{
  trg_t *trg = cnt->trg;
  uint64_t i;
  char out[16]; /* 4 byte sample nr + 8 char trg code */

  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &cnt->evt, FOURCC_evt, &cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &cnt->evt, FOURCC_evt, &cnt->cnt), CNTERR_FILE);
  }
  for (i = 0; i < trg->c; i++) {
    if(cnt->mode==CNT_RIFF) {
      swrite_s32(out, trg->v[i].sample);
      strncpy(&out[4], trg->v[i].code, 8);
      RET_ON_RIFFERROR(riff_write(out, 12, 1, cnt->f, &cnt->evt), CNTERR_FILE);
    } else {
      swrite_u64(out, trg->v[i].sample);
      strncpy(&out[8], trg->v[i].code, 8);
      RET_ON_RIFFERROR(riff64_write(out, 16, 1, cnt->f, &cnt->evt), CNTERR_FILE);
    }
  }
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_close(cnt->f, cnt->evt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_close(cnt->f, cnt->evt), CNTERR_FILE);
  }
  return CNTERR_NONE;
}

int write_tfh_chunk(eeg_t *cnt)
{
  var_string textbuf;
  int retcode;
  /* Create the TFH chunk */
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &cnt->tfh, FOURCC_tfh, &cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &cnt->tfh, FOURCC_tfh, &cnt->cnt), CNTERR_FILE);
  }
  textbuf = varstr_construct(); /* Create header string */
  if (NULL == textbuf)
    return CNTERR_MEM;
  if(cnt->mode==CNT_RIFF) {
    writehead_TFH(&cnt->tf_header, textbuf); /* Fill header string */
    retcode = riff_write(varstr_cstr(textbuf), varstr_length(textbuf), 1, cnt->f, &cnt->tfh); /* Write header to chunk */
  } else {
    // TODO, unsupported time freq data
    writehead_TFH(&cnt->tf_header, textbuf); /* Fill header string */
    retcode = riff64_write(varstr_cstr(textbuf), varstr_length(textbuf), 1, cnt->f, &cnt->tfh); /* Write header to chunk */
  }
  varstr_destruct(textbuf); /* Destroy header string */
  if (RIFFERR_NONE != retcode) /* Return on writing error */
  return CNTERR_FILE;

  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_close(cnt->f, cnt->tfh), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_close(cnt->f, cnt->tfh), CNTERR_FILE);
  }
  return CNTERR_NONE;
}

int write_recinfo_chunk(eeg_t *cnt, record_info_t* recinfo)
{
  var_string textbuf;
  int retcode;
  char line[512];
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &cnt->info, FOURCC_info, &cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &cnt->info, FOURCC_info, &cnt->cnt), CNTERR_FILE);
  }

  textbuf = varstr_construct();
  if (NULL == textbuf)
    return CNTERR_MEM;

  sprintf(line, "[StartDate]\n%.20le\n", recinfo->m_startDate);
  varstr_append(textbuf, line);
  sprintf(line, "[StartFraction]\n%.20le\n", recinfo->m_startFraction);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[Hospital]\n%s\n", recinfo->m_szHospital);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[TestName]\n%s\n", recinfo->m_szTestName);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[TestSerial]\n%s\n", recinfo->m_szTestSerial);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[Physician]\n%s\n", recinfo->m_szPhysician);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[Technician]\n%s\n", recinfo->m_szTechnician);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[MachineMake]\n%s\n", recinfo->m_szMachineMake);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[MachineModel]\n%s\n", recinfo->m_szMachineModel);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[MachineSN]\n%s\n", recinfo->m_szMachineSN);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectName]\n%s\n", recinfo->m_szName);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectID]\n%s\n", recinfo->m_szID);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectAddress]\n%s\n", recinfo->m_szAddress);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectPhone]\n%s\n", recinfo->m_szPhone);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectSex]\n%c\n", recinfo->m_chSex);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectDateOfBirth]\n%d %d %d %d %d %d %d %d %d\n",
          recinfo->m_DOB.tm_sec, recinfo->m_DOB.tm_min, recinfo->m_DOB.tm_hour,
          recinfo->m_DOB.tm_mday, recinfo->m_DOB.tm_mon, recinfo->m_DOB.tm_year,
          recinfo->m_DOB.tm_wday, recinfo->m_DOB.tm_yday, recinfo->m_DOB.tm_isdst);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[SubjectHandedness]\n%c\n", recinfo->m_chHandedness);
  varstr_append(textbuf, line);
  snprintf(line, 512, "[Comment]\n%s\n", recinfo->m_szComment);
  varstr_append(textbuf, line);

  if(cnt->mode==CNT_RIFF) {
    retcode = riff_write(varstr_cstr(textbuf), varstr_length(textbuf), 1, cnt->f, &cnt->info);
  } else {
    retcode = riff64_write(varstr_cstr(textbuf), varstr_length(textbuf), 1, cnt->f, &cnt->info);
  }
  varstr_destruct(textbuf);
  if (RIFFERR_NONE != retcode) /* Return on writing error */
    return CNTERR_FILE;

  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_close(cnt->f, cnt->info), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_close(cnt->f, cnt->info), CNTERR_FILE);
  }
  return CNTERR_NONE;
}

int read_recinfo_chunk(eeg_t *cnt, record_info_t* recinfo) {
  int this_chunk_contains_binary_data=1;
  uint64_t file_position;

  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_open(cnt->f, &cnt->info, FOURCC_info, cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_open(cnt->f, &cnt->info, FOURCC_info, cnt->cnt), CNTERR_FILE);
  }

  memset((void*)recinfo, 0, sizeof(record_info_t));

  file_position=eepio_ftell(cnt->f);
  // Parse new-style ASCII header
  {
    FILE *f = cnt->f;
    int nread = 0;
    char line[256];
    double dbl = -1.0;

    do {
      fgets(line, 256, f); nread += strlen(line);

      if(line[0]==0) {
        break;
      }

      if (*line == '[') {
        if (strstr(line, "[StartDate]")) {
          fgets(line, 256, f); nread += strlen(line);
          if (sscanf(line, "%le", &dbl) != 1) {
            return 1;
          }
          recinfo->m_startDate = dbl;
          // hack around a mistake for not writing version-info in files.
          // if we read ascii and successfully find a StartDate in ascii, then surely
          // this file does not a contain binary chunk.
          this_chunk_contains_binary_data=0;
        }
        else if (strstr(line, "[StartFraction]")) {
          fgets(line, 256, f); nread += strlen(line);
          if (sscanf(line, "%le", &dbl) != 1)
            return 1;
          recinfo->m_startFraction = dbl;
        }
        else if (strstr(line, "[SubjectSex]")) {
          fgets(line, 256, f); nread += strlen(line);
          sscanf(line, "%c", &recinfo->m_chSex);
        }
        else if (strstr(line, "[SubjectHandedness]")) {
          fgets(line, 256, f); nread += strlen(line);
          sscanf(line, "%c", &recinfo->m_chHandedness);
        }
        else if (strstr(line, "[SubjectDateOfBirth]")) {
          fgets(line, 256, f); nread += strlen(line);
          if (sscanf(line, "%d %d %d %d %d %d %d %d %d",
                    &recinfo->m_DOB.tm_sec, &recinfo->m_DOB.tm_min,
                    &recinfo->m_DOB.tm_hour, &recinfo->m_DOB.tm_mday,
                    &recinfo->m_DOB.tm_mon, &recinfo->m_DOB.tm_year,
                    &recinfo->m_DOB.tm_wday, &recinfo->m_DOB.tm_yday,
                    &recinfo->m_DOB.tm_isdst) != 9)
              return 1;
        }
        nread += match_config_str(f, line, "[Hospital]", recinfo->m_szHospital, 256);
        nread += match_config_str(f, line, "[TestName]", recinfo->m_szTestName, 256);
        nread += match_config_str(f, line, "[TestSerial]", recinfo->m_szTestSerial, 256);
        nread += match_config_str(f, line, "[Physician]", recinfo->m_szPhysician, 256);
        nread += match_config_str(f, line, "[Technician]", recinfo->m_szTechnician, 256);
        nread += match_config_str(f, line, "[MachineMake]", recinfo->m_szMachineMake, 256);
        nread += match_config_str(f, line, "[MachineModel]", recinfo->m_szMachineModel, 256);
        nread += match_config_str(f, line, "[MachineSN]", recinfo->m_szMachineSN, 256);
        nread += match_config_str(f, line, "[SubjectName]", recinfo->m_szName, 256);
        nread += match_config_str(f, line, "[SubjectID]", recinfo->m_szID, 256);
        nread += match_config_str(f, line, "[SubjectAddress]", recinfo->m_szAddress, 256);
        nread += match_config_str(f, line, "[SubjectPhone]", recinfo->m_szPhone, 256);
        nread += match_config_str(f, line, "[Comment]", recinfo->m_szComment, 256);
      }
    } while (nread < cnt->info.size);

//  return ferror(f);
  }
  if(this_chunk_contains_binary_data) {
    // rewind file to earlier saved postition
    eepio_fseek(cnt->f, file_position, SEEK_SET);
    // Unfortunately, there exist CNT files without version numbers, which might have
    // recording info in binary format. We only read the 'start_time' here (first few bytes);
    if(cnt->mode==CNT_RIFF) {
      RET_ON_RIFFERROR(riff_read((char *) recinfo, sizeof(double), 2, cnt->f, cnt->info), CNTERR_FILE);
    } else {
      RET_ON_RIFFERROR(riff64_read((char *) recinfo, sizeof(double), 2, cnt->f, cnt->info), CNTERR_FILE);
    }
    recinfo->m_startDate=eep_byteswap_8_double_safe(recinfo->m_startDate);
    recinfo->m_startFraction=eep_byteswap_8_double_safe(recinfo->m_startFraction);
    // The other fields will stay empty
  }
//printf("##### %s(%i) %s\n", __FILE__, __LINE__, __FUNCTION__);
//printf("##### recording info was read from %s data\n", this_chunk_contains_binary_data ? "BINARY" : "ASCII");
//printf("##### rec_inf.m_startDate: %f\n", recinfo->m_startDate);
//printf("##### rec_inf.m_startFraction: %f\n", recinfo->m_startFraction);
  return CNTERR_NONE;
}

int eep_finish_file(eeg_t *EEG)
{
  FILE *f, *fin;
  long i;
  char *code;
  short val, chanc;
  uint64_t sample;
  int flag, oldflag;
  unsigned long  pos;

  if (!EEG)
    return CNTERR_NONE;

  f = EEG->f;
  chanc = EEG->eep_header.chanc;
  switch (EEG->mode) {

    case CNT_NS30:
      if (EEG->store[DATATYPE_EEG].data.writeflag) {
        return CNTERR_DATA;
      }
      break;

    case CNT_EEP20:
      if (EEG->store[DATATYPE_EEG].data.writeflag) {
        char tag[128];
        /* dump into NS binary header */
        if (puthead_EEP20(EEG)) return CNTERR_FILE;
        strcpy(tag, TAG_EEP20);
        strcat(tag, "  ");
        if (eepio_fseek(f, 0, SEEK_SET)) return CNTERR_FILE;
        if (!eepio_fwrite(tag, strlen(tag)-1, 1, f)) return CNTERR_FILE;

        /* mangle trigger table into data area */

        /* need to read the file to conserve present flags */
        if (EEG->fname && (fin = eepio_fopen(EEG->fname, "rb"))) {
          fflush(f);
          for (i = 0; i < trg_get_c(EEG->trg); i++) {
            code = trg_get(EEG->trg, i, &sample);
            flag = 0;
            if (TRG_IS_DCRESET(code))
              flag |= EEP20_DCRESET;
            if (TRG_IS_DISCONT(code))
              flag |= EEP20_DISCONT;
            if (sscanf(code, "%hd", &val))
              flag |= (val & 0xff);
            pos = 900 + 75 * chanc + sample * (chanc + 2) * 2 + chanc * 2;

            if (    eepio_fseek(fin, pos, SEEK_SET)
                || !read_s16(fin, &oldflag) )  return CNTERR_FILE;
            flag |= (oldflag & 0xff00);
            if (    eepio_fseek(f, pos, SEEK_SET)
                || !write_s16(f, flag) ) return CNTERR_FILE;
          }
          eepio_fclose(fin);
        }

      }
      break;

    case CNT_RIFF:
    case CNTX_RIFF:
      if (!EEG->finalized)
        for (i=0; i < NUM_DATATYPES; i++)
          if (EEG->store[i].data.writeflag)
          {
            make_partial_output_consistent(EEG, 1 /* Finalize data */);
            break;
          }
      break;

    default:
      return CNTERR_DATA;
  }
  eep_free(EEG);
  return CNTERR_NONE;

}

int eep_fclose(eeg_t* cnt)
{
  int status;
  FILE* file = cnt->f;
  status = eep_finish_file(cnt);
  if(file)
    eepio_fclose(file);
  return status;
}

int eep_seek(eeg_t *EEG, eep_datatype_e type, uint64_t s, int relative)
{
  int r = CNTERR_NONE;
  uint64_t newpos;

  switch (EEG->mode) {
    case CNT_NS30:
      if (relative)
        newpos = EEG->store[DATATYPE_EEG].data.bufepoch * EEG->store[DATATYPE_EEG].epochs.epochl + EEG->store[DATATYPE_EEG].data.readpos + s;
      else
        newpos = s;
      if (newpos / EEG->store[DATATYPE_EEG].epochs.epochl != EEG->store[DATATYPE_EEG].data.bufepoch)
        r = getepoch_NS30(EEG, newpos / EEG->store[DATATYPE_EEG].epochs.epochl);
      EEG->store[DATATYPE_EEG].data.readpos = newpos % EEG->store[DATATYPE_EEG].epochs.epochl;
      break;

    case CNT_EEP20:
      if (eepio_fseek(EEG->f,  SAMPLESTART_EEP20(EEG->eep_header.chanc)
                      + s * SAMPLESIZE_EEP20(EEG->eep_header.chanc), relative ? SEEK_CUR : SEEK_SET))
        r = CNTERR_FILE;
      if (relative)
        EEG->store[DATATYPE_EEG].data.readpos += s;
      else
        EEG->store[DATATYPE_EEG].data.readpos = s;
      break;

    case CNT_AVR:
      if (relative)
        EEG->store[type].data.readpos+= s;
      else
        EEG->store[type].data.readpos = s;
      break;

    case CNT_RIFF:
    case CNTX_RIFF:
      r = eep_seek_impl(EEG, type, s, relative);
      break;

    default:
      r = CNTERR_DATA;
  }

  return r;
}

int init_data_store(eeg_t *cnt, eep_datatype_e type)
{
  chunk_t   dummychunk;
  chunk64_t dummychunk64;
  uint64_t chanseq_len = 0;
  storage_t *store = &cnt->store[type];

  chanseq_len = cnt->eep_header.chanc;
  if (DATATYPE_TIMEFREQ == type)
    chanseq_len *= 2 * cnt->tf_header.componentc;

  /* Open top-level chunk */
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_list_open(cnt->f, &dummychunk, store->fourcc, cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_list_open(cnt->f, &dummychunk64, store->fourcc, cnt->cnt), CNTERR_FILE);
  }

  /* read chunk channel sequence */
  RET_ON_CNTERROR(read_chanseq_chunk(cnt, store, chanseq_len));

  /* read epoch table */
  RET_ON_CNTERROR(read_epoch_chunk(cnt, store));

  switch (type)
  {
    case DATATYPE_EEG:
      /* compression buffer setup */
      RET_ON_CNTERROR(cnt_create_raw3_compr_buffer(cnt));
      break;

    case DATATYPE_TIMEFREQ:
      store->data.buf_float = (float*)
       v_malloc((size_t) TF_CNTBUF_SIZE(cnt, store->epochs.epochl), "td_data.buf");

      store->data.cbuf = (char*)
       v_malloc((size_t) TF_CBUF_SIZE(cnt, store->epochs.epochl), "td_data.cbuf");
      break;

    case DATATYPE_AVERAGE:
    case DATATYPE_STDDEV:
      store->data.buf_float = (float*)
       v_malloc((size_t) FLOAT_CNTBUF_SIZE(cnt, store->epochs.epochl), "rawfdata.buf");

      store->data.cbuf = (char*)
       v_malloc((size_t) FLOAT_CBUF_SIZE(cnt, store->epochs.epochl), "rawfdata.cbuf");
      break;

    default:
      return CNTERR_BADREQ;
  }

  /* open data area and fill first buffer */
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_open(cnt->f, &store->ch_data, FOURCC_data, store->ch_toplevel), CNTERR_DATA);
  } else {
    RET_ON_RIFFERROR(riff64_open(cnt->f, &store->ch_data, FOURCC_data, store->ch_toplevel), CNTERR_DATA);
  }

#ifdef CNT_MMAP
  RET_ON_CNTERROR(mmap_data_chunk(cnt->f, store));
#endif
  store->initialized = 1;

  return getepoch_impl(cnt, type, 0);
}

int eep_read_sraw (eeg_t *cnt, eep_datatype_e type, sraw_t *muxbuf, uint64_t n)
{
  uint64_t i;
  FILE *f     = cnt->f;
  storage_t *store = &cnt->store[DATATYPE_EEG];
  short chanc = cnt->eep_header.chanc;
  long step = (long)chanc;
  size_t bbuf = (size_t) step * sizeof(sraw_t);
  int state;
  double scale;
  short chan;
  sraw_t *srcstart, *dststart;
  sraw_t statusflags[2];

  switch (cnt->mode) {
    case CNT_NS30:
      for (i = 0; i < n; i++)
      {
        srcstart = &store->data.buf_int[cnt->store[DATATYPE_EEG].data.readpos];
        dststart = &muxbuf[i * step];

        /* multiplex the NS block */
        for (chan = 0; chan < chanc; chan++)
          dststart[chan] = srcstart[chan * store->epochs.epochl];

        store->data.readpos++;
        if (store->data.readpos == store->epochs.epochl) {
          if (store->data.bufepoch < store->epochs.epochc - 1) {
            if ((state = getepoch_NS30(cnt, store->data.bufepoch + 1))) {
              return state;
            }
          }
          else {
            store->data.readpos = 0;
            store->data.bufepoch++;
          }
        }
      }
      return CNTERR_NONE;

    case CNT_EEP20:
      for (i = 0; i < n; i++)
      {
        if (vread_s16(f, &muxbuf[i*step], step) != step)
          return CNTERR_FILE;
        if (vread_s16(f, statusflags, 2) != 2)
          return CNTERR_FILE;

        if (statusflags[0] & EEP20_TRGMASK)
          trg_set_EEP20(cnt->trg, store->data.readpos, statusflags[0]);

        store->data.readpos ++;
      }

      return CNTERR_NONE;

    case CNT_RIFF:
    case CNTX_RIFF:
      if (type != DATATYPE_EEG) {
        return CNTERR_BADREQ; /* All other types have float data */
      }
      if (!store->initialized) {
         return CNTERR_DATA; /* No such data in this file */
      }
      if (store->data.readpos + store->data.bufepoch * store->epochs.epochl + n > eep_get_samplec(cnt)) {
          return CNTERR_RANGE; /* Sample out of range */
      }
      for (i = 0; i < n; i++)
      {
        /* 1 sample per channel + 4 bytes control to 0 */
        memcpy(&muxbuf[i * step], &store->data.buf_int[store->data.readpos * step], bbuf);

        store->data.readpos++;
        if (store->data.readpos == store->epochs.epochl) {
          /* can we read a next buffer ? */
          if (store->data.bufepoch < store->epochs.epochc - 1) {
            if ((state = getepoch_impl(cnt, type, store->data.bufepoch + 1))) {
              return state;
            }
          }
          /* or increment counters only (needed for eep_seek) */
          else {
            store->data.readpos = 0;
            store->data.bufepoch++;
          }
        }
      }
      return CNTERR_NONE;

    case CNT_AVR:
      if (type != DATATYPE_AVERAGE)
        return CNTERR_BADREQ;

      store = &cnt->store[type];
      if (!store->initialized)
         return CNTERR_DATA; /* No such data in this file */

      for (chan = 0; chan < chanc; chan++)
      {
        scale = eep_get_chan_scale(cnt, chan);
        for (i = 0; i < n; i++)
          muxbuf[i*step+chan] = (sraw_t) FRND(store->data.buf_float[(store->data.readpos + i)*step + chan] / scale);
      }
      store->data.readpos+=n;
      return CNTERR_NONE;

    default:
      return CNTERR_DATA;
  }
}

int eep_write_sraw (eeg_t *cnt, const sraw_t *muxbuf, uint64_t n)
{
  long step = cnt->eep_header.chanc;
  size_t outbytes = (size_t) step * sizeof(sraw_t);
  storage_t *store;

  uint64_t i;

  switch (cnt->mode) {
    case CNT_EEP20:
      return CNTERR_BADREQ; /* Use eep_write_sraw_EEP20 instead */

    case CNT_RIFF:
    case CNTX_RIFF:
      if ((DATATYPE_EEG != cnt->current_datachunk) || (!cnt->store[cnt->current_datachunk].initialized)) {
        return CNTERR_BADREQ; /* No RAW3 data or chunk not initialized */
      }
      store = &cnt->store[cnt->current_datachunk];
      for (i = 0; i < n; i++)
      {
        memcpy(&store->data.buf_int[store->data.writepos * step], &muxbuf[i * step], outbytes);
        (store->data.writepos)++;
        if (store->data.writepos == store->epochs.epochl)
        {
          if (putepoch_impl(cnt)) {
            return CNTERR_FILE;
          }
        }
      }

      /* force a call to make_partial_output_consistent after first sample written.
         otherwise we have to wait until the first full epoch is written, which may
         take a second. We don't have this second when reading the CNT file in real-
         time simultaneously */
      if (cnt->keep_consistent) {
          uint64_t sc;
          eep_get_samplec_full(cnt, & sc);
          if(sc == 1) {
              // fprintf(stderr, "making consistent, sc=%i\n", (int)sc);
              return make_partial_output_consistent(cnt, 0 /* No finalize */);
          }
      }
      return CNTERR_NONE;

    default:
      return CNTERR_DATA;
  }
}


/* accessible eeg_t members ------------------------------------------- */
char *eep_get_name(eeg_t *cnt)
{
  return cnt->fname;
}

int eep_get_mode(eeg_t *cnt)
{
  return cnt->mode;
}

double eep_get_period(eeg_t *cnt)
{
  return cnt->eep_header.period;
}

short eep_get_rate(eeg_t *cnt)
{
  return (short) ((double) 1.0 / cnt->eep_header.period + 0.5);
}

void eep_set_period(eeg_t *cnt, double period)
{
  short rate;
  cnt->eep_header.period = period;

  /* raw3 epoch length */
  if (cnt->mode == CNT_RIFF || cnt->mode == CNTX_RIFF) {
    cnt->store[DATATYPE_EEG].epochs.epochl = MIN(eep_get_rate(cnt), 100);
  }

  /* EEP 2.0 stores an integer sampling rate only - sync the period here */
  else
  {
    rate = (short) (1.0 / period + 0.5);
    cnt->eep_header.period = (float) 1.0 / rate;
  }
}

short eep_get_chanc(eeg_t *cnt)
{
  return cnt->eep_header.chanc;
}

int  eep_get_chan_index(eeg_t *cnt, const char *lab)
{
  int i = 0;

  while (i < cnt->eep_header.chanc && strcasecmp(cnt->eep_header.chanv[i].lab, lab)) i++;

  if (i == cnt->eep_header.chanc)
    return -1;
  else
    return i;
}


char *eep_get_chan_label(eeg_t *cnt, short chan)
{
  return cnt->eep_header.chanv[chan].lab;
}

void  eep_set_chan_label(eeg_t *cnt, short chan, const char *lab)
{
  strncpy(cnt->eep_header.chanv[chan].lab, lab, 10);
  cnt->eep_header.chanv[chan].lab[9] = '\0';
}

double eep_get_chan_scale(eeg_t *cnt, short chan)
{
  return (cnt->eep_header.chanv[chan].iscale * cnt->eep_header.chanv[chan].rscale);
}

void  eep_set_chan_scale(eeg_t *cnt, short chan, double scale)
{
  cnt->eep_header.chanv[chan].rscale = scale / cnt->eep_header.chanv[chan].iscale;
}

char *eep_get_chan_unit(eeg_t *cnt, short chan)
{
  return cnt->eep_header.chanv[chan].runit;
}

eegchan_t get_chan(eeg_t *cnt, short chan)
{
  return cnt->eep_header.chanv[chan];
}

uint64_t eep_get_samplec(eeg_t *cnt)
{
  return cnt->eep_header.samplec;
}

int eep_get_samplec_full(const eeg_t *cnt, uint64_t *samplec)
{
	const storage_t * store = NULL;

	if ((DATATYPE_UNDEFINED == cnt->current_datachunk) ||
		(!cnt->store[cnt->current_datachunk].initialized))
		return CNTERR_BADREQ;

	store = &cnt->store[cnt->current_datachunk];

	(*samplec) = store->data.writepos + cnt->eep_header.samplec;
	return CNTERR_NONE;
}

/* manipulation of cnt structure internals */

void eep_set_mode_EEP20(eeg_t *cnt)
{
  short rate;

  cnt->mode = CNT_EEP20;

  v_free(cnt->store[DATATYPE_EEG].chanseq);
  cnt->store[DATATYPE_EEG].epochs.epochl = 0;

  /* EEP 2.0 stores an integer sampling rate only - sync the period here */
  rate = (short) (1.0 / cnt->eep_header.period + 0.5);
  cnt->eep_header.period = (double) 1.0 / rate;
}

eegchan_t *eep_chan_init(short chanc)
{
  eegchan_t *chanv;

  if(chanc < 1 || chanc > CNT_MAX_CHANC)
    return NULL;

  chanv = (eegchan_t *) v_malloc(chanc * sizeof(eegchan_t), "chanv");
  if (NULL != chanv)
    memset((void*)chanv, 0, sizeof(eegchan_t));
  return chanv;
}

void eep_chan_set(eegchan_t *chanv, short chan,
                  const char *lab, double iscale, double rscale, const char *runit)
{
  eegchan_t *c = &(chanv[chan]);
  strncpy(c->lab, lab, 10);
  c->lab[9] = 0;
  c->iscale = iscale;
  c->rscale = rscale;
  strncpy(c->runit, runit, 10);
  c->runit[9] = 0;
  c->reflab[0] = 0;
  c->status[0] = 0;
  c->type[0] = 0;
}

void eep_chan_set_reflab(eegchan_t *chanv, short chan, const char *reflab)
{
  eegchan_t *c = &(chanv[chan]);
  strncpy(c->reflab, reflab, 10);
  c->reflab[9] = '\0';
}

trg_t *eep_get_trg(eeg_t *cnt)
{
  return cnt->trg;
}

void eep_set_trg(eeg_t *cnt, trg_t *trg)
{

  if( cnt->trg == trg ) return;

  trg_free(cnt->trg);
  cnt->trg = trg;

}

void eep_dup_chan(eeg_t *cnt, short chan, char *newlab)
{
  eegchan_t *s, *d;
  short* new_seq;
  int comp;
  int type;

  /* in this case we cannot increment further the counter which holds the number of channels */
  if(cnt->eep_header.chanc == CNT_MAX_CHANC)
  {
    /* TODO: indicate error? */
    return;
  }

  cnt->eep_header.chanv = (eegchan_t *) v_realloc(cnt->eep_header.chanv, (cnt->eep_header.chanc + 1) * sizeof(eegchan_t), "chanv");
  s = &cnt->eep_header.chanv[chan];
  d = &cnt->eep_header.chanv[cnt->eep_header.chanc];
  strcpy(d->runit, s->runit);
  d->rscale = s->rscale;
  d->iscale = s->iscale;
  strncpy(d->lab, newlab, 10); d->lab[9] = '\0';

  //if (CNT_TFRAW3 == cnt->mode || CNT_TF == cnt->mode)
  for (type = 0; type < NUM_DATATYPES; type++)
  {
    storage_t *store = &cnt->store[type];
    switch (type)
    {
      case DATATYPE_TIMEFREQ:
        if (store->initialized)
        {
          new_seq = (short*)
           v_malloc(2 * (cnt->eep_header.chanc + 1) * cnt->tf_header.componentc * sizeof(short), "tf_chanseq");

           for (comp = 0; comp < cnt->tf_header.componentc; comp++)
           {
             /* Copy all existing channel values for this component */
             memcpy(&new_seq[2 * comp * (cnt->eep_header.chanc + 1)],
                &store->chanseq[2 * comp * cnt->eep_header.chanc],
                2 * cnt->eep_header.chanc);
            /* Now add the new sequence number as (comp, chanc) */
            new_seq[2 * (comp + 1) * (cnt->eep_header.chanc + 1) - 2] = comp;
            new_seq[2 * (comp + 1) * (cnt->eep_header.chanc + 1) - 1] = cnt->eep_header.chanc;
          }
          v_free(store->chanseq);
          store->chanseq = new_seq;
       }
       break;

     case DATATYPE_EEG:
     case DATATYPE_AVERAGE:
     case DATATYPE_STDDEV:
       if (store->initialized)
       {
         store->chanseq = (short *) v_realloc(store->chanseq, (cnt->eep_header.chanc + 1) * sizeof(short), "chanseq");
         store->chanseq[cnt->eep_header.chanc] = cnt->eep_header.chanc;
       }
     }
   }
   cnt->eep_header.chanc++;
}

int eep_switch_to_write(eeg_t *cnt, eep_datatype_e type)
{
  storage_t *store = &cnt->store[type];
  uint64_t chanseq_len = cnt->eep_header.chanc;
  if (DATATYPE_TIMEFREQ == type)
    chanseq_len *= 2 * cnt->tf_header.componentc;
  if (DATATYPE_UNDEFINED != cnt->current_datachunk)
    RET_ON_CNTERROR(close_data_chunk(cnt, 0, &cnt->store[cnt->current_datachunk]));
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_list_new(cnt->f, &store->ch_toplevel, store->fourcc, &cnt->cnt), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_list_new(cnt->f, &store->ch_toplevel, store->fourcc, &cnt->cnt), CNTERR_FILE);
  }
  RET_ON_CNTERROR(write_chanseq_chunk(cnt, store, chanseq_len));
  if(cnt->mode==CNT_RIFF) {
    RET_ON_RIFFERROR(riff_new(cnt->f, &store->ch_data, FOURCC_data, &store->ch_toplevel), CNTERR_FILE);
  } else {
    RET_ON_RIFFERROR(riff64_new(cnt->f, &store->ch_data, FOURCC_data, &store->ch_toplevel), CNTERR_FILE);
  }
  store->data.writeflag = 1;

  cnt->current_datachunk = type;
  return CNTERR_NONE;
}

int eep_prepare_to_write(eeg_t *cnt, eep_datatype_e type, uint64_t epochl, short *chanv)
{
  storage_t *store = &cnt->store[type];
  uint64_t seq_len = cnt->eep_header.chanc;
  uint64_t chan, comp, seq = 0;
  if (type == DATATYPE_TIMEFREQ)
    seq_len *= cnt->tf_header.componentc;

  eep_clear_epochs(cnt, store);
  store->epochs.epochl = epochl;

  v_free(store->chanseq);
  switch (type)
  {
    case DATATYPE_EEG:
    case DATATYPE_STDDEV:
    case DATATYPE_AVERAGE:
      cnt->eep_header.samplec = 0;
      store->chanseq = (short *) v_malloc(seq_len * sizeof(short), "chanseq");
      if (NULL == store->chanseq)
        return CNTERR_MEM;
      for (chan = 0; chan < seq_len; chan++)
        store->chanseq[chan] = chanv ? chanv[chan] : chan;
      break;

    case DATATYPE_TIMEFREQ:
      cnt->tf_header.samplec = 0;
      store->chanseq = (short *) v_malloc(2 * seq_len * sizeof(short), "chanseq");
      if (NULL == store->chanseq)
        return CNTERR_MEM;
      for (comp = 0; comp < cnt->tf_header.componentc; comp++)
        for (chan = 0; chan < cnt->eep_header.chanc; chan++)
        {
          store->chanseq[seq] = chanv ? chanv[seq] : comp;
          seq++;
          store->chanseq[seq] = chanv ? chanv[seq] : chan;
          seq++;
        }
      break;

    default:
      return CNTERR_BADREQ;
  }

  switch (type)
  {
    case DATATYPE_EEG:
      raw3_free(cnt->r3);
      RET_ON_CNTERROR(cnt_create_raw3_compr_buffer(cnt));
      if (NULL == cnt->r3)
        return CNTERR_MEM;
      break;

    case DATATYPE_STDDEV:
    case DATATYPE_AVERAGE:
      v_free(store->data.buf_float);
      store->data.buf_float = (float*)
       v_malloc((size_t) FLOAT_CNTBUF_SIZE(cnt, store->epochs.epochl), "data.buf_float");

      v_free(store->data.cbuf);
      store->data.cbuf = (char*)
       v_malloc((size_t) FLOAT_CBUF_SIZE(cnt, store->epochs.epochl), "data.cbuf");
      break;

    case DATATYPE_TIMEFREQ:
      v_free(store->data.buf_float);
      store->data.buf_float = (float*)
        v_malloc((size_t) TF_CNTBUF_SIZE(cnt, store->epochs.epochl), "data.buf_float");

      v_free(store->data.cbuf);
      store->data.cbuf = (char*)
       v_malloc((size_t) TF_CBUF_SIZE(cnt, store->epochs.epochl), "data.cbuf");
      break;

    default:
      return CNTERR_BADREQ;
  }
  store->initialized = 1;

  return eep_switch_to_write(cnt, type); /* New behavior: immediately switch to writing */
}

/* Newly added interface functions for Time/Frequency data */
void eep_comp_set(tf_component_t *compv, short comp, float value, const char *descr)
{
  tf_component_t *tfc = &(compv[comp]);
  strncpy(tfc->description, descr, 40);
  tfc->description[39] = '\0';
  tfc->axis_value = value;
}

tf_component_t* eep_comp_init(short compc)
{
  tf_component_t *compv = (tf_component_t *) v_malloc(compc * sizeof(tf_component_t), "comp");
  return compv;
}

int eep_get_compc(eeg_t *cnt)
{
  return cnt->tf_header.componentc;
}

int eep_dup_comp(eeg_t *cnt, short comp, float newvalue)
{
  tf_header_t *tfhead = &cnt->tf_header;
  storage_t *store = &cnt->store[DATATYPE_TIMEFREQ];
  tf_component_t *s, *d;
  int chan;

  if (!store->initialized)
    return CNTERR_DATA;

  tfhead->componentv = (tf_component_t *)
    v_realloc(tfhead->componentv, (tfhead->componentc + 1) * sizeof(tf_component_t), "componentv");
  s = &tfhead->componentv[comp];
  d = &tfhead->componentv[tfhead->componentc];
  strncpy(d->description, s->description, 40);
  d->axis_value = newvalue;

  store->chanseq = (short *) v_realloc(store->chanseq, 2 * cnt->eep_header.chanc * (tfhead->componentc+1) * sizeof(short), "tf_chanseq");
  for (chan = 0; chan < cnt->eep_header.chanc; chan++)
  {
    store->chanseq[2 * tfhead->componentc + chan] = comp;
    store->chanseq[2 * tfhead->componentc + chan + 1] = chan;
  }
  tfhead->componentc++;

  return CNTERR_NONE;
}

int eep_get_comp_index(eeg_t *cnt, float value)
{
  int i;
  tf_header_t *tfhead = &cnt->tf_header;

  for (i = 0; i < tfhead->componentc; i++)
    if (tfhead->componentv[i].axis_value == value)
      return i;

  return -1;
}

float eep_get_comp_value(eeg_t *cnt, int comp_index)
{
  if (comp_index < 0 || comp_index >= cnt->tf_header.componentc)
    return -1.0;

  return cnt->tf_header.componentv[comp_index].axis_value;
}

int eep_set_comp_value(eeg_t *cnt, int comp_index, float value)
{
  if (comp_index < 0 || comp_index >= cnt->tf_header.componentc)
    return CNTERR_DATA;

  cnt->tf_header.componentv[comp_index].axis_value = value;
  return CNTERR_NONE;
}

const char* eep_get_comp_description(eeg_t *cnt, int comp_index)
{
  if (comp_index < 0 || comp_index >= cnt->tf_header.componentc)
    return NULL;

  return cnt->tf_header.componentv[comp_index].description;
}

int eep_set_comp_description(eeg_t *cnt, int comp_index, const char* description)
{
  if (comp_index < 0 || comp_index >= cnt->tf_header.componentc)
    return CNTERR_DATA;

  strncpy(cnt->tf_header.componentv[comp_index].description, description, 40);
  cnt->tf_header.componentv[comp_index].description[39] = '\0'; /* For non-C99-compliant strncpy */
    return CNTERR_NONE;
}

const char *eep_get_comp_axis_unit(eeg_t *cnt)
{
  return cnt->tf_header.tf_unit;
}

void eep_set_comp_axis_unit(eeg_t *cnt, const char *unit)
{
  strncpy(cnt->tf_header.tf_unit, unit, 16);
  cnt->tf_header.tf_unit[15] = '\0';
}

const char* eep_get_tf_type(eeg_t *cnt)
{
  return cnt->tf_header.tf_type;
}

void eep_set_tf_type(eeg_t *cnt, const char *tf_type)
{
  strncpy(cnt->tf_header.tf_type, tf_type, 40);
  cnt->tf_header.tf_type[39] = '\0';
}

tf_content_e eep_get_tf_contenttype(eeg_t *cnt)
{
  return cnt->tf_header.content_datatype;
}

void eep_set_tf_contenttype(eeg_t *cnt, tf_content_e tf_contenttype)
{
  cnt->tf_header.content_datatype = tf_contenttype;
}

double eep_get_tf_period(eeg_t *cnt)
{
  return cnt->tf_header.period;
}

void eep_set_tf_period(eeg_t *cnt, double period)
{
  cnt->tf_header.period = period;
}

short eep_get_tf_rate(eeg_t *cnt)
{
  return (short) (1.0 / cnt->tf_header.period);
}

uint64_t eep_get_tf_samplec(eeg_t *cnt)
{
  return cnt->tf_header.samplec;
}

const char* eep_get_tf_chan_unit(eeg_t *cnt, int chan_index)
{
  /* This is tricky to implement without using (or 90% duplicating) the code in
  utDataUnitManager */
  /* Maybe it would be good to have a (C-implementation of the) dataunitmanager? */
  /* Have implemented this in ASA for now, see: ioCNTEEGDataFileImport_c::GetTFDataUnit() */

  return NULL; /* Not implemented yet */
}


int eep_has_recording_info(eeg_t *cnt)
{
  return (NULL != cnt->recording_info);
}

void eep_set_recording_info(eeg_t *cnt, record_info_t* info) {
  if(info) {
    if(!cnt->recording_info) {
      cnt->recording_info = (record_info_t*) v_malloc(sizeof(record_info_t) , "recording_info");
    }
    memcpy(cnt->recording_info, info, sizeof(record_info_t));
  } else {
    if(cnt->recording_info) {
      v_free(cnt->recording_info);
      cnt->recording_info = NULL;
    }
  }
}

void eep_get_recording_info(eeg_t *cnt, record_info_t* info) {
  if (NULL != cnt->recording_info) {
    memcpy(info, cnt->recording_info, sizeof(record_info_t));
  } else {
    memset(info, 0, sizeof(record_info_t));
  }
}

/* Private helpers for make_partial_output_consistent.
Decreases the size of the specified chunks parents with the specified amount. */
int decrease_chunksize(FILE* f, chunk_t* chunk, uint64_t to_subtract, int is_cnt_riff)
{
  uint64_t filepos;
  chunk_t* x;
  if (0 == to_subtract)
    return CNTERR_NONE;

  filepos = eepio_ftell(f);
  x = chunk->parent;
  while (NULL != x)
  {
      x->size -= to_subtract;
      eepio_fseek(f, x->start, SEEK_SET);
      if(is_cnt_riff) {
        RET_ON_RIFFERROR(riff_put_chunk(f, *x), CNTERR_FILE);
      } else {
        RET_ON_RIFFERROR(riff64_put_chunk(f, *x), CNTERR_FILE);
      }
    x = x->parent;
  }
  eepio_fseek(f, filepos, SEEK_SET);
  return CNTERR_NONE;
}

/* Closes a data chunk and keeps track of current/previous chunk sizes */
int close_data_chunk(eeg_t *cnt, int finalize, storage_t *store /*, chunk_mode_e write_mode*/)
{
    if (finalize)
      RET_ON_CNTERROR(putepoch_impl(cnt));
    if(cnt->mode==CNT_RIFF) {
      RET_ON_RIFFERROR(riff_close(cnt->f, store->ch_data), CNTERR_FILE);
    } else {
      RET_ON_RIFFERROR(riff64_close(cnt->f, store->ch_data), CNTERR_FILE);
    }
    /* write epochtable */
    RET_ON_CNTERROR(write_epoch_chunk(cnt, store));

    /* RIFF tricks: because we might be closing the same chunks several times (and then
       overwriting them later), we have to subtract the 'previous' chunk sizes (e.g. as they
       where during the previous cal of make_partial_output_consistent() ). */
    RET_ON_CNTERROR(decrease_chunksize(cnt->f, &store->ch_data, store->data_size, cnt->mode==CNT_RIFF));
    /* Note that the '+8' is missing in the previous line because the 'data'
       chunk won't be reopened, just truncated */

    store->data_size = store->ch_data.size;
    store->data_size+= store->data_size & 1; /* Make sure it's even! */
    if (store->ep_size > 0) {
      if(cnt->mode==CNT_RIFF) {
        RET_ON_CNTERROR(decrease_chunksize(cnt->f, &store->ch_ep, store->ep_size + 8, 1));
      } else {
        RET_ON_CNTERROR(decrease_chunksize(cnt->f, &store->ch_ep, store->ep_size + 12, 0));
      }
    }
    store->ep_size = store->ch_ep.size;
    store->ep_size+= store->ep_size & 1; /* Make sure it's even! */
    return CNTERR_NONE;
}
/*
* Explanation of this odd-looking code:
*  The idea is that the CNT file is brought into a consistent state even when the file is
*  not yet finished. This makes it more friendly to crashes and power failures during acquisition.
*
*  The RIFF format is hierarchical; every chunk stores the (added) size of all it's children.
*  When you call riff_close(some_chunk) it will update the size of all its parents (e.g. adding
*  it's own "final" size to that of the parents).
*  However, here we might call riff_close() on the same chunk several times.
*  So, each time we call this riff_close, we have to manually adjust the chunk sizes of it's parents,
*  or these would become much too big (every chunk that's closed twice would be counted
*  double in it's parents size!). So we subtract the 'previous' size when we last closed this
*  chunk from all its parents (see decrease_chunksize()).
*
*  When you're wondering about the extra 8 bytes: that is the size of a RIFF chunk header. If
*  the chunk is completely rewritten (this happens with EEPH but not with DATA, for example),
*  you also have to subtract this from the parent size (see riff_new() and you'll understand).
*
*  About the 'even' sizes: even though chunks can have odd sizes, their actual size in the file
*  has to be even (this is a restriction 'by design' in the RIFF file format). riff_close() will
*  write an extra junk-byte if the size is odd. So, even though the chunks own size would be odd,
*  it will add the extra byte to the size of the parents, hence we have to subtract it again as well.
*
*  This code is ugly because it messes around with RIFF internals, but I cannot think of a better
*  way except fundamentally changing the 'riff' handling code (which would be time-consuming and
*  error-prone).
*    - WH20040105
*/

int make_partial_output_consistent(eeg_t *cnt, int finalize)
{
  FILE* f = cnt->f;

  uint64_t file_offset;

  /* Remember the current file position */
  uint64_t filepos = eepio_ftell(f);

  /* Do everything that eep_finish_file() normally takes care of, such as closing the
    active data chunk, writing the corresponding EPochs chunk, the header, etc.
  */
  if (DATATYPE_UNDEFINED != cnt->current_datachunk)
    close_data_chunk(cnt, finalize, &cnt->store[cnt->current_datachunk]);
  
  /* Write the Recording info chunk, if one is specified */
  if (NULL != cnt->recording_info) {
    write_recinfo_chunk(cnt, cnt->recording_info);
  }

  /* create and write the ascii header */
  RET_ON_CNTERROR(write_eeph_chunk(cnt));
  cnt->eep_header.chunk_size = cnt->eeph.size;
  cnt->eep_header.chunk_size += cnt->eep_header.chunk_size & 1; /* Make sure it's even! */

  /* When TF data has been written add a TFH chunk */
  if (cnt->store[DATATYPE_TIMEFREQ].initialized)
  {
    RET_ON_CNTERROR(write_tfh_chunk(cnt));
    cnt->tf_header.chunk_size = cnt->tfh.size;
    cnt->tf_header.chunk_size += cnt->tf_header.chunk_size & 1; /* Make sure it's even! */
  }

  /* dump trigger archive if present (only when finalizing) */
  if ( finalize )
  {
    if( cnt->trg && (cnt->trg->c > 0) )
      write_trigger_chunk(cnt);
  }

  // correct CNT chunk
  file_offset = eepio_ftell(f);
  eepio_fseek(f, cnt->cnt.start, SEEK_SET);
  if(cnt->mode==CNT_RIFF) {
    cnt->cnt.size = file_offset - 8;
    RET_ON_RIFFERROR(riff_put_chunk(f, cnt->cnt), CNTERR_FILE);
  } else {
    cnt->cnt.size = file_offset - 12;
    RET_ON_RIFFERROR(riff64_put_chunk(f, cnt->cnt), CNTERR_FILE);
  }

  /* Finally, restore the file pointer to it's original position */
  if (!finalize)
    eepio_fseek(f, filepos, SEEK_SET);
  cnt->finalized = finalize;
  return CNTERR_NONE;
}

double eep_get_chan_iscale(eeg_t *cnt,  short chan)
{
  return cnt->eep_header.chanv[chan].iscale;
}

double eep_get_chan_rscale(eeg_t *cnt,  short chan)
{
  return cnt->eep_header.chanv[chan].rscale;
}

void eep_set_chan_iscale(eeg_t *cnt,  short chan, double scale)
{
  cnt->eep_header.chanv[chan].iscale = scale;
}

void eep_set_chan_rscale(eeg_t *cnt,  short chan, double scale)
{
  cnt->eep_header.chanv[chan].rscale = scale;
}

void eep_set_chan_reflab(eeg_t *cnt, short chan, const char *reflab)
{
  strncpy(cnt->eep_header.chanv[chan].reflab, reflab, 10);
  cnt->eep_header.chanv[chan].reflab[9] = '\0';
}

void eep_set_chan_unit(eeg_t* cnt, short chan, const char* unit)
{
  strncpy(cnt->eep_header.chanv[chan].runit, unit, 16);
  cnt->eep_header.chanv[chan].runit[15] = '\0';
}

char* eep_get_chan_reflab(eeg_t *cnt, short chan)
{
  return cnt->eep_header.chanv[chan].reflab;
}

void eep_set_chan_status(eeg_t *cnt, short chan, const char *status)
{
  strncpy(cnt->eep_header.chanv[chan].status, status, 10);
  cnt->eep_header.chanv[chan].status[9] = '\0';
}

char* eep_get_chan_status(eeg_t *cnt, short chan)
{
  return cnt->eep_header.chanv[chan].status;
}

void eep_set_chan_type(eeg_t *cnt, short chan, const char *type)
{
  strncpy(cnt->eep_header.chanv[chan].type, type, 10);
  cnt->eep_header.chanv[chan].type[9] = '\0';
}

char* eep_get_chan_type(eeg_t *cnt, short chan)
{
  return cnt->eep_header.chanv[chan].type;
}

int eep_get_epochl(eeg_t *cnt, eep_datatype_e type)
{
  if (cnt->store[type].initialized)
    return cnt->store[type].epochs.epochl;
  return -1;
}

int eep_has_data_of_type(eeg_t *cnt, eep_datatype_e type)
{
  return cnt->store[type].initialized;
}

short* eep_get_chanseq(eeg_t *cnt, eep_datatype_e type)
{
  storage_t* store = &cnt->store[type];
  short* result = NULL;
  uint64_t size = cnt->eep_header.chanc * sizeof(short);
  if (type == DATATYPE_TIMEFREQ)
    size *= 2 * cnt->tf_header.componentc;

  if (store->initialized)
  {
    result = (short*) v_malloc(size, "chanseq");
    memcpy(result, store->chanseq, size);
  }
  return result;
}

int eep_get_neuroscan_type(eeg_t *cnt)
{
  return cnt->ns_cnttype;
}

void eep_get_fileversion(eeg_t *cnt, char *version)
{
  sprintf(version, "%d.%d", cnt->eep_header.fileversion_major, cnt->eep_header.fileversion_minor);
}

int eep_get_fileversion_major(eeg_t *cnt)
{
  return cnt->eep_header.fileversion_major;
}

int eep_get_fileversion_minor(eeg_t *cnt)
{
  return cnt->eep_header.fileversion_minor;
}

void eep_get_dataformat(eeg_t *cnt, char *format)
{
  int mode = eep_get_mode(cnt);

  switch (mode) {
    case CNT_NS30:
      if( eep_get_neuroscan_type(cnt) == 3 )
        strcpy(format, "NeuroScan 3.x (16 bit blocked)");
      if( eep_get_neuroscan_type(cnt) == 1 )
        strcpy(format, "NeuroScan 4.1 (16 bit channel multiplexed)");
      break;
    case CNT_EEP20:
      strcpy(format, "EEP 2.0 (16 bit channel multiplexed)"); break;
    case CNT_RIFF:
    case CNTX_RIFF:
      if( eep_get_fileversion_major(cnt) )
        sprintf(format, "EEP %d.%d", eep_get_fileversion_major(cnt), eep_get_fileversion_minor(cnt));
      else
        sprintf(format, "EEP 3.x");
      if( mode == CNTX_RIFF )
        strcat(format, " (64-bit RIFF variant)");
      if( eep_has_data_of_type(cnt, DATATYPE_EEG) )
        strcat(format, " (32 bit raw3 compressed)");
      if( eep_has_data_of_type(cnt, DATATYPE_TIMEFREQ) )
        strcat(format, " time-frequency");
      if( eep_has_data_of_type(cnt, DATATYPE_AVERAGE) )
        strcat(format, " average");
      if( eep_has_data_of_type(cnt, DATATYPE_STDDEV) )
        strcat(format, " stddev");
      if( eep_has_data_of_type(cnt, DATATYPE_TIMEFREQ) ||
          eep_has_data_of_type(cnt, DATATYPE_AVERAGE)  ||
          eep_has_data_of_type(cnt, DATATYPE_STDDEV) )
        strcat(format, " (float vectors)");
      break;
    case CNT_AVR:
      strcpy(format, "EEP 2.0/3.x avr (float vectors)"); break;
    default:
      strcpy(format, "unknown");
  }
}

int eep_has_history(eeg_t *cnt) {
  return (cnt->history != NULL);
}

void eep_set_history(eeg_t *cnt, const char *hist)
{
  if (NULL == cnt->history) {
    cnt->history = varstr_construct();
  }
  if(hist==NULL) {
    hist="no history";
  }
  varstr_set(cnt->history, hist);
}

void eep_append_history(eeg_t *cnt, const char *histline)
{
  if (NULL == cnt->history)
    cnt->history = varstr_construct();

  if (varstr_length(cnt->history) > 0)
    varstr_append(cnt->history, "\n");

  varstr_append(cnt->history, histline);
}

const char* eep_get_history(eeg_t *cnt)
{
  return varstr_cstr(cnt->history);
}

int eep_read_float(eeg_t *cnt, eep_datatype_e type, float *muxbuf, uint64_t n)
{
  uint64_t i;
  short chanc = cnt->eep_header.chanc;
  long step = chanc;
  storage_t *store = &cnt->store[type];

  // todo
  if (CNT_RIFF != cnt->mode && CNTX_RIFF != cnt->mode && CNT_AVR != cnt->mode) {
    return CNTERR_BADREQ;
  }

  if (!store->initialized)
    return CNTERR_DATA; /* No such data in this file */

  if (store->data.readpos + store->data.bufepoch * store->epochs.epochl +
      n > (DATATYPE_TIMEFREQ == type ? eep_get_tf_samplec(cnt) : eep_get_samplec(cnt)))
    return CNTERR_RANGE;

  switch (type)
  {
    case DATATYPE_TIMEFREQ:
      step *= cnt->tf_header.componentc;
      /* Deliberately falling through here */

    case DATATYPE_AVERAGE:
    case DATATYPE_STDDEV:
      for (i = 0; i < n; i++) /* Handle one sample */
      {
        /* Copy the data for 1 sample of the currently loaded epoch */
        memcpy(&muxbuf[i * step], &store->data.buf_float[store->data.readpos * step], step * sizeof(float));
        store->data.readpos++;
        if (CNT_AVR != cnt->mode && /* old style AVR's have no epochs */
            store->data.readpos == store->epochs.epochl) /* Need to load next epoch? */
        { /* Yes, try to load next epoch */
          if (store->data.bufepoch < store->epochs.epochc - 1) /* Are there any more epochs? */
          { /* Yes, so go ahead and load it */
            RET_ON_CNTERROR(getepoch_impl(cnt, type, store->data.bufepoch + 1));
          }
          else
          { /* There are no more epochs to read. Just update the counters (for eep_seek) */
            store->data.readpos = 0;
            store->data.bufepoch++;
          }
        }
      }
      break;

    default:
      return CNTERR_BADREQ;
  }

  return CNTERR_NONE;
}

int eep_write_float(eeg_t *cnt, float *muxbuf, uint64_t n)
{
  uint64_t i;
  short chanc = cnt->eep_header.chanc;
  long step = chanc;
  storage_t *store;

  if (cnt->mode != CNT_RIFF && cnt->mode != CNTX_RIFF) {
    return CNTERR_BADREQ;
  }

  if ((DATATYPE_UNDEFINED == cnt->current_datachunk) ||
      (DATATYPE_EEG == cnt->current_datachunk) || /* can't write RAW3 data */
      (!cnt->store[cnt->current_datachunk].initialized)) {
      return CNTERR_BADREQ;
  }

  store = &cnt->store[cnt->current_datachunk];

  switch (cnt->current_datachunk)
  {
    case DATATYPE_TIMEFREQ:
      step *= cnt->tf_header.componentc;
      /* Deliberately falling through */

    case DATATYPE_AVERAGE:
    case DATATYPE_STDDEV:
      for (i = 0; i < n; i++)
      {
        memcpy(&store->data.buf_float[store->data.writepos * step], &muxbuf[i * step], step * sizeof(float));
        store->data.writepos++;
        if (store->data.writepos == store->epochs.epochl) {
          RET_ON_CNTERROR(putepoch_impl(cnt));
        }
      }
      break;

    default:
      return CNTERR_BADREQ;
  }
  return CNTERR_NONE;
}

long eep_get_total_trials(eeg_t *cnt)
{
  return cnt->eep_header.total_trials;
}

long eep_get_averaged_trials(eeg_t *cnt)
{
  return cnt->eep_header.averaged_trials;
}

const char* eep_get_conditionlabel(eeg_t *cnt)
{
  return cnt->eep_header.conditionlabel;
}

const char* eep_get_conditioncolor(eeg_t *cnt)
{
  return cnt->eep_header.conditioncolor;
}

double eep_get_pre_stimulus_interval(eeg_t *cnt)
{
  return cnt->eep_header.pre_stimulus;
}

void eep_set_total_trials(eeg_t *cnt, long total_trials)
{
  cnt->eep_header.total_trials = total_trials;
}

void eep_set_averaged_trials(eeg_t *cnt, long averaged_trials)
{
  cnt->eep_header.averaged_trials = averaged_trials;
}

void eep_set_conditionlabel(eeg_t *cnt, const char* conditionlabel)
{
  strncpy(cnt->eep_header.conditionlabel, conditionlabel, 25);
  cnt->eep_header.conditionlabel[24] = '\0';
}

void eep_set_conditioncolor(eeg_t *cnt, const char* conditioncolor)
{
  strncpy(cnt->eep_header.conditioncolor, conditioncolor, 25);
  cnt->eep_header.conditioncolor[24] = '\0';
}

void eep_set_pre_stimulus_interval(eeg_t *cnt, double pre_stimulus)
{
  cnt->eep_header.pre_stimulus = pre_stimulus;
}

/*****************************************************************************/
/*********************** Recording time/date routines ************************/
/*
 some info:
 please visit http://www.minelinks.com/calendar_converter.html for more information
 the offset(diff between 1jan1970 and 30dec1899) is 2209161600( 70 years in seconds)
 *****************************************************************************/

void
eep_exceldate_to_unixdate(double excel, double fraction, time_t *epoch) {
  time_t return_value=0;

  // 27538 -> 1970jan1
  // 2958464 -> 30dec9999
  if( excel >= 27538 && excel <= 2958464 ) {
    return_value=(time_t)(round(excel*3600.0*24.0)-2209161600);
  }
  (*epoch) = return_value;
}

void
eep_unixdate_to_exceldate(time_t epoch, double *excel, double *fraction) {
  double return_value=0;

  return_value=(double)(epoch + 2209161600) / (3600.0*24.0);

  (*excel) = return_value;
  (*fraction) = 0;
}

/*****************************************************************************/
/*********************** time/date helper functions **************************/
/*****************************************************************************/

time_t
eep_get_time_epoch(double t, double f) {
  time_t l=0;

  eep_exceldate_to_unixdate(t, f, &l);

  return l;
}

void eep_get_time_string(double t, double f, char *s) {
  struct tm time;
  eep_get_time_struct(t, f, &time);
  strcpy(s, asctime(&time));
}

void eep_get_time_struct(double t, double f, struct tm *tm) {
  time_t epoch;

  epoch=eep_get_time_epoch(t, f);
  memcpy(tm, localtime(&epoch), sizeof(struct tm));
}

/*****************************************************************************/
/*********************** Recording time/date routines ***********************/
/*****************************************************************************/

time_t
eep_get_recording_startdate_epoch(eeg_t *cnt) {
  time_t l=0;
  record_info_t rec_inf;

  if(eep_has_recording_info(cnt)) {
    eep_get_recording_info(cnt, &rec_inf);
    l = eep_get_time_epoch(rec_inf.m_startDate, rec_inf.m_startFraction);
  }

  return l;
}

void eep_get_recording_startdate_string(eeg_t *cnt, char *s) {
  struct tm recording_time;
  eep_get_recording_startdate_struct(cnt, &recording_time);
  strcpy(s, asctime(&recording_time));
}

void eep_get_recording_startdate_struct(eeg_t *cnt, struct tm *tm) {
  time_t epoch;

  epoch=eep_get_recording_startdate_epoch(cnt);
  memcpy(tm, localtime(&epoch), sizeof(struct tm));
}

void eep_set_recording_startdate_epoch(eeg_t *cnt, time_t epoch) {
  record_info_t rec_inf;

  if(eep_has_recording_info(cnt)) {
    eep_get_recording_info(cnt, &rec_inf);
    eep_unixdate_to_exceldate(epoch, &rec_inf.m_startDate, &rec_inf.m_startFraction);
    eep_set_recording_info(cnt, &rec_inf);
  }
}

void eep_set_recording_startdate_struct(eeg_t *cnt, struct tm *tm) {
  time_t epoch;
  record_info_t rec_inf;

  epoch=mktime(tm);
  if(eep_has_recording_info(cnt)) {
    eep_get_recording_info(cnt, &rec_inf);
    eep_set_recording_startdate_epoch(cnt, epoch);
  }
}

const char *eep_get_hospital(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szHospital;
	}
	return NULL;
}

const char *eep_get_test_name(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szTestName;
	}
	return NULL;
}

const char *eep_get_test_serial(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szTestSerial;
	}
	return NULL;
}

const char *eep_get_physician(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szPhysician;
	}
	return NULL;
}

const char *eep_get_technician(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szTechnician;
	}
	return NULL;
}

const char *eep_get_machine_make(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szMachineMake;
	}
	return NULL;
}

const char *eep_get_machine_model(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szMachineModel;
	}
	return NULL;
}

const char *eep_get_machine_serial_number(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szMachineSN;
	}
	return NULL;
}

const char *eep_get_patient_name(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szName;
	}
	return NULL;
}

const char *eep_get_patient_id(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szID;
	}
	return NULL;
}

const char *eep_get_patient_address(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szAddress;
	}
	return NULL;
}

const char *eep_get_patient_phone(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szPhone;
	}
	return NULL;
}

char eep_get_patient_sex(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_chSex;
	}
	return 0;
}

char eep_get_patient_handedness(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_chHandedness;
	}
	return 0;
}

struct tm *eep_get_patient_day_of_birth(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return &(cnt->recording_info->m_DOB);
	}
	return NULL;
}

const char *eep_get_comment(eeg_t *cnt) {
	if (eep_has_recording_info(cnt)) {
		return cnt->recording_info->m_szComment;
	}
	return NULL;
}

void eep_set_hospital(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szHospital) / sizeof(cnt->recording_info->m_szHospital[0]) - 1;
		strncpy(cnt->recording_info->m_szHospital, value, len);
	}
}

void eep_set_test_name(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szTestName) / sizeof(cnt->recording_info->m_szTestName[0]) - 1;
		strncpy(cnt->recording_info->m_szTestName, value, len);
	}
}

void eep_set_test_serial(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szTestSerial) / sizeof(cnt->recording_info->m_szTestSerial[0]) - 1;
		strncpy(cnt->recording_info->m_szTestSerial, value, len);
	}
}

void eep_set_physician(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szPhysician) / sizeof(cnt->recording_info->m_szPhysician[0]) - 1;
		strncpy(cnt->recording_info->m_szPhysician, value, len);
	}
}

void eep_set_technician(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szTechnician) / sizeof(cnt->recording_info->m_szTechnician[0]) - 1;
		strncpy(cnt->recording_info->m_szTechnician, value, len);
	}
}

void eep_set_machine_make(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szMachineMake) / sizeof(cnt->recording_info->m_szMachineMake[0]) - 1;
		strncpy(cnt->recording_info->m_szMachineMake, value, len);
	}
}

void eep_set_machine_model(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szMachineModel) / sizeof(cnt->recording_info->m_szMachineModel[0]) - 1;
		strncpy(cnt->recording_info->m_szMachineModel, value, len);
	}
}

void eep_set_machine_serial_number(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szMachineSN) / sizeof(cnt->recording_info->m_szMachineSN[0]) - 1;
		strncpy(cnt->recording_info->m_szMachineSN, value, len);
	}
}

void eep_set_patient_name(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szName) / sizeof(cnt->recording_info->m_szName[0]) - 1;
		strncpy(cnt->recording_info->m_szName, value, len);
	}
}

void eep_set_patient_id(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szID) / sizeof(cnt->recording_info->m_szID[0]) - 1;
		strncpy(cnt->recording_info->m_szID, value, len);
	}
}

void eep_set_patient_address(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szAddress) / sizeof(cnt->recording_info->m_szAddress[0]) - 1;
		strncpy(cnt->recording_info->m_szAddress, value, len);
	}
}

void eep_set_patient_phone(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szPhone) / sizeof(cnt->recording_info->m_szPhone[0]) - 1;
		strncpy(cnt->recording_info->m_szPhone, value, len);
	}
}

void eep_set_patient_sex(eeg_t *cnt, char value) {
	if (eep_has_recording_info(cnt)) {
		cnt->recording_info->m_chSex = value;
	}
}

void eep_set_patient_handedness(eeg_t *cnt, char value) {
	if (eep_has_recording_info(cnt)) {
		cnt->recording_info->m_chHandedness = value;
	}
}

void eep_set_patient_day_of_birth(eeg_t *cnt, struct tm *value) {
	if (eep_has_recording_info(cnt) && value) {
		cnt->recording_info->m_DOB = *value;
	}
}

void eep_set_comment(eeg_t *cnt, const char *value) {
	if (eep_has_recording_info(cnt) && value) {
		const size_t len = sizeof(cnt->recording_info->m_szComment) / sizeof(cnt->recording_info->m_szComment[0]) - 1;
		strncpy(cnt->recording_info->m_szComment, value, len);
	}
}

/*****************************************************************************/
/*********************** Backwards compatibility stuff ***********************/
/*****************************************************************************/

/* header read funcs --------------------------------------------- */

void fix_label_NS30(char * lab, size_t len)
{
  char *s_p = &lab[0];
  while (*s_p != '\0' && (s_p-lab) < len) {
    if (isspace((int) *s_p)) *s_p = '_';
    s_p++;
  }
}

int getchanhead_NS30(eeg_t *EEG, FILE *f, int chan)
{
  float      sens, calib;
  eegchan_t  *elec = &(EEG->eep_header.chanv[chan]);
  const long ofs_elec = GENHEADER_SIZE + chan * CHANHEADER_SIZE;

  eepio_fseek(f, ofs_elec + OFS_LAB, SEEK_SET);
  eepio_fread((char *) elec->lab, sizeof(elec->lab), 1, f);
  fix_label_NS30(elec->lab,sizeof(elec->lab));

  eepio_fseek(f, ofs_elec + OFS_SENS, SEEK_SET);
  read_f32(f, &sens);

  eepio_fseek(f, ofs_elec + OFS_CALIB, SEEK_SET);
  read_f32(f, &calib);
  elec->iscale = sens * calib;
  elec->rscale = RSCALE_NS30;
  strcpy(elec->runit, "uV");

  return ferror(f);
}

int gethead_NS30(eeg_t *EEG)
{
  FILE *f = EEG->f;
  int in;
  short chan;

  eepio_fseek(f, OFS_CNTTYPE, SEEK_SET);
  eepio_fread(&EEG->ns_cnttype,1,1,f);
  if(EEG->ns_cnttype != 1 && EEG->ns_cnttype != 3 )
    eeperror("unknown NS cnt type (%d)!\n", EEG->ns_cnttype);

  eepio_fseek(f, OFS_NCHAN, SEEK_SET);
  read_s16(f, &in);
  EEG->eep_header.chanc = in;

  if (EEG->eep_header.chanc < 1 || EEG->eep_header.chanc > CNT_MAX_CHANC) return 1;
  EEG->eep_header.chanv = (eegchan_t *) v_malloc(EEG->eep_header.chanc * sizeof(eegchan_t),"chanv");

  for(chan = 0; chan < EEG->eep_header.chanc; chan++)
    getchanhead_NS30(EEG, f, chan);

  eepio_fseek(f, OFS_RATE, SEEK_SET);
  read_s16(f, &in);
  EEG->eep_header.period = (double) 1.0 / in;

  /*
  eepio_fseek(f, NSOFS_SAMPLEC, SEEK_SET);
  read_dos_long(f, &EEG->eep_header.samplec);
  */

  /* get position of event table */
  eepio_fseek(f, OFS_EVTPOS, SEEK_SET);
  read_s32(f, &EEG->ns_evtpos);

  /*derive number of samples in the cnt */
  EEG->eep_header.samplec = (EEG->ns_evtpos - 900 -75 * EEG->eep_header.chanc) / (2*EEG->eep_header.chanc);

  if (eepio_fseek(f, EEG->ns_evtpos, SEEK_SET)) return 1;
  eepio_fread(&EEG->ns_evttype,1,1,f);
  switch (EEG->ns_evttype) {
    case 1: EEG->ns_evtlen =  8; break;
    case 2: EEG->ns_evtlen = 19; break;
    default: eepstatus("unknown event type! event table ignored!\n");
            EEG->ns_evtc = 0;
      EEG->ns_evtlen = 0;
      break;
  }
  if(EEG->ns_evtlen) {
    read_s32(f, &in);
    EEG->ns_evtc = in / EEG->ns_evtlen;
    read_s32(f, &in);
    EEG->ns_evtpos += 9 + in;
  }
  eepio_fseek(f, OFS_BLOCKL, SEEK_SET);

  {
    int32_t itmp;
    read_s32(f, &itmp);
    EEG->store[DATATYPE_EEG].epochs.epochl = itmp;
  }

  if (EEG->store[DATATYPE_EEG].epochs.epochl > 1 && EEG->ns_cnttype==3)  /* blocked format */
    EEG->store[DATATYPE_EEG].epochs.epochl /= 2;
  else
    EEG->store[DATATYPE_EEG].epochs.epochl = 1; /* multiplexed format */

  EEG->store[DATATYPE_EEG].epochs.epochc = EEG->eep_header.samplec / EEG->store[DATATYPE_EEG].epochs.epochl;
  EEG->eep_header.samplec = EEG->store[DATATYPE_EEG].epochs.epochc * EEG->store[DATATYPE_EEG].epochs.epochl;

  return ferror(f);
}


int gethead_AVR(eeg_t *EEG)
{
  avr_t avr;
  int chan, i;

  float *v, max, scale;

  FILE *f = EEG->f;

  if( avropen( &avr, f) != AVRERR_NONE )
      return CNTERR_DATA;

  EEG->eep_header.chanc = (short) avr.chanc;
  EEG->eep_header.samplec = (slen_t) avr.samplec;
  EEG->eep_header.period = (double) avr.period; /* sampling interval in sec */
  EEG->eep_header.total_trials = avr.trialc;
  EEG->eep_header.averaged_trials = avr.trialc - avr.rejtrialc;
  EEG->eep_header.pre_stimulus = -1.0 * (float)avr.sample0 * avr.period;
  strncpy(EEG->eep_header.conditionlabel, avr.condlab, 25);
  EEG->eep_header.conditionlabel[24] = '\0';
  strncpy(EEG->eep_header.conditioncolor, avr.condcol, 25);
  EEG->eep_header.conditioncolor[24] = '\0';

  eep_set_history(EEG, "");
  for (i=0; i < avr.histc; i++)
    eep_append_history(EEG, avr.histv[i]);

  trg_set(EEG->trg, (long) -avr.sample0, "t0"); /* produce a stim. offset trigger */

  /* prepare channel table */
  EEG->eep_header.chanv = (eegchan_t *)
    v_malloc(EEG->eep_header.chanc * sizeof(eegchan_t), "chanv");
  memset(EEG->eep_header.chanv, 0, EEG->eep_header.chanc * sizeof(eegchan_t));

  for (chan = 0; chan < EEG->eep_header.chanc; chan++) {
    sscanf(avr.chanv[chan].lab, "%10s", EEG->eep_header.chanv[chan].lab);
    /*strcpy(EEG->chanv[chan].lab, avr.chanv[chan].lab);*/
    strcpy(EEG->eep_header.chanv[chan].runit, "uV");
    EEG->eep_header.chanv[chan].rscale = 1.0;
  }

  v = (float *) v_malloc(EEG->eep_header.samplec * sizeof(float),"v");
  EEG->store[DATATYPE_AVERAGE].data.buf_float = (float *) v_malloc(EEG->eep_header.samplec * EEG->eep_header.chanc * sizeof(float), "buf");
  EEG->store[DATATYPE_AVERAGE].data.bufepoch = 0;
  EEG->store[DATATYPE_AVERAGE].epochs.epochl = (slen_t) avr.samplec; /* Everything in 1 epoch */

  for (chan = 0; chan < EEG->eep_header.chanc; chan++)
  {
    if (AVRERR_NONE != avrseek(&avr, f, chan, AVRBAND_MEAN) ||
        AVRERR_NONE != avrread(f, v, EEG->eep_header.samplec))
      return CNTERR_FILE;

    /* Prepare scaling per channel (for converting to 32-bit INT) */
    /* jw: Why would we ever want to do that??? It seems like really old code... */
    /* TODO: check use of this code */
    max = -1.0;
    for (i = 0; i < EEG->eep_header.samplec; i++)
      if (fabs(v[i]) > max)
        max = (float) fabs(v[i]);
    scale = (float) (max > 0.0 ? (float) (1 << 30) / max : 1.0);
    EEG->eep_header.chanv[chan].iscale = 1.0 / scale;

    for (i = 0; i < EEG->eep_header.samplec; i++)
      EEG->store[DATATYPE_AVERAGE].data.buf_float[chan + i * EEG->eep_header.chanc] = v[i];

    if (AVRERR_NONE == avrseek( &avr, f, chan, AVRBAND_VAR))
    {
      if (!EEG->store[DATATYPE_STDDEV].initialized)
      {
        EEG->store[DATATYPE_STDDEV].data.buf_float = (float*) v_malloc(EEG->eep_header.samplec * EEG->eep_header.chanc * sizeof(float), "buf");
        EEG->store[DATATYPE_STDDEV].initialized = 1;
      }
      if (AVRERR_NONE != avrread( f, v, EEG->eep_header.samplec))
        return CNTERR_FILE;

      for (i = 0; i < EEG->eep_header.samplec; i++)
        EEG->store[DATATYPE_STDDEV].data.buf_float[chan + i * EEG->eep_header.chanc] = (float) sqrt(v[i]);
    }
  }
  EEG->store[DATATYPE_AVERAGE].initialized = 1;

  free(v);
  avrclose(&avr);

  return ferror(f);
}



/* data read funcs ---------------------------------------------- */

int getepoch_NS30(eeg_t *EEG, uint64_t epoch)
{
  /* need to seek source file position ? */
  if (EEG->store[DATATYPE_EEG].data.bufepoch != epoch - 1) {
    if (eepio_fseek(EEG->f,
          SAMPLESTART_EEP20(EEG->eep_header.chanc)
        + epoch * EEG->eep_header.chanc * EEG->store[DATATYPE_EEG].epochs.epochl * 2, SEEK_SET))
      return CNTERR_FILE;
  }

  /* get the 1 block of data in buffer */
  if (   vread_s16(EEG->f, EEG->store[DATATYPE_EEG].data.buf_int, EEG->eep_header.chanc * EEG->store[DATATYPE_EEG].epochs.epochl)
      != EEG->eep_header.chanc * EEG->store[DATATYPE_EEG].epochs.epochl) {
    return CNTERR_FILE;
  }

  EEG->store[DATATYPE_EEG].data.bufepoch = epoch;
  EEG->store[DATATYPE_EEG].data.readpos = 0;

  return CNTERR_NONE;
}

int getchanhead_EEP20(eeg_t *EEG, int chan)
{
  float scale_EEP20;

  eegchan_t  *elec = &(EEG->eep_header.chanv[chan]);
  const long ofs_elec = GENHEADER_SIZE + chan * CHANHEADER_SIZE;

  eepio_fseek(EEG->f, ofs_elec + OFS_LAB, SEEK_SET);
  eepio_fread((char *) elec->lab, sizeof(elec->lab)-1, 1, EEG->f);
  elec->lab[10] = 0;

  eepio_fseek(EEG->f, ofs_elec + OFS_CALIB, SEEK_SET);
  read_f32(EEG->f, &scale_EEP20);
  elec->iscale = (double) scale_EEP20;
  elec->rscale = RSCALE_EEP20;
  strcpy(elec->runit, "uV");

  return ferror(EEG->f);
}

int gethead_EEP20(eeg_t *EEG)
{
  int in;
  int chan;

  eepio_fseek(EEG->f, OFS_NCHAN, SEEK_SET);
  read_s16(EEG->f, &in);
  EEG->eep_header.chanc = in;
  if (EEG->eep_header.chanc < 0 || EEG->eep_header.chanc > 1024) return 1; /* EEG->eep_header.chanc shall be less than 1024? and not less then CNT_MAX_CHANC? */

  EEG->eep_header.chanv = (eegchan_t *) v_malloc(EEG->eep_header.chanc * sizeof(eegchan_t), "chanv");
  for(chan = 0; chan < EEG->eep_header.chanc; chan++)
    getchanhead_EEP20(EEG, chan);

  eepio_fseek(EEG->f, OFS_RATE, SEEK_SET);
  read_s16(EEG->f, &in);
  EEG->eep_header.period = (double) 1.0 / in;

  return ferror(EEG->f);
}

int putchanhead_EEP20(eeg_t *EEG, int chan)
{
  /* force the correct real world conversion */

  eegchan_t elec = EEG->eep_header.chanv[chan];
  float eepscale = (float) ((double) elec.iscale * elec.rscale / RSCALE_EEP20);

  const long ofs_elec = GENHEADER_SIZE + chan * CHANHEADER_SIZE;

  eepio_fseek(EEG->f, ofs_elec + OFS_LAB, SEEK_SET);
  eepio_fwrite((char *) elec.lab, strlen(elec.lab) + 1, 1, EEG->f);

  eepio_fseek(EEG->f, ofs_elec + OFS_CALIB, SEEK_SET);

  write_f32(EEG->f, eepscale);

  return ferror(EEG->f);
}

int puthead_EEP20(eeg_t *EEG)
{
  int chan;

  eepio_fseek(EEG->f, 0, SEEK_SET);
  eepio_fwrite(TAG_EEP20, strlen(TAG_EEP20), 1, EEG->f);

  eepio_fseek(EEG->f, OFS_NCHAN, SEEK_SET);
  write_s16(EEG->f, EEG->eep_header.chanc);

  eepio_fseek(EEG->f, OFS_RATE, SEEK_SET);
  write_s16(EEG->f, (int) (1.0 / EEG->eep_header.period + 0.5));

  for(chan = 0; chan < EEG->eep_header.chanc; chan++)
    putchanhead_EEP20(EEG, chan);

  return ferror(EEG->f);
}

int trg_read_NS30(eeg_t *EEG)
{
  FILE  *f    = EEG->f;
  trg_t *trg  = EEG->trg;
  short chanc = EEG->eep_header.chanc;

  long i;
  slen_t sample;
  char code[TRG_CODE_LENGTH + 1];
  slen_t rsshift;

  /* NeuroScan trigger data */
  int evt;
  int evtr;
  int evtpos;


  switch( EEG->ns_cnttype ) {
    case 3: /* 7/100 sec reset record delay */
            rsshift = (slen_t) (0.07 / EEG->eep_header.period + 0.5);
      break;
    case 1: /* 8/100 sec DC correct delay */
            rsshift = (slen_t) (0.08 / EEG->eep_header.period + 0.5);
      break;

    default: eeperror("unknonw NS cnt type (%d)\n", EEG->ns_cnttype);
  }


  if (eepio_fseek(f, EEG->ns_evtpos, SEEK_SET))
    return CNTERR_DATA;

  trg_set(trg, 0, TRG_DISCONT);

  for (i = 0; i < EEG->ns_evtc; i++) {
    read_u16(f, &evt);
    read_u16(f, &evtr);
    read_s32(f, &evtpos);
    /*skip rest of event entry */
    if(EEG->ns_evtlen) eepio_fseek(f,EEG->ns_evtlen - 8, SEEK_CUR);
    if (ferror(f))
      return CNTERR_FILE;

    /* convert the NeuroScan event table to EEP trg table */
    sample = (evtpos - 900 - 75 * chanc) / (chanc * 2);

    if (sample < EEG->eep_header.samplec) {
      if ((evtr & 0xf000) == 0xb000)
        trg_set(trg, sample + rsshift, TRG_DCRESET);

      if ((evtr & 0xf000) == 0xe000)
        trg_set(trg, sample, TRG_DISCONT);

      if (evt & 0xff) {
        sprintf(code, "%d", evt & 0xff);
        trg_set(trg, sample, code);
      }
    }
  }
  return CNTERR_NONE;
}

int cntopen_NS30(eeg_t *EEG)
{
  int r;
  EEG->mode = CNT_NS30;
  EEG->trg = trg_init();


  if (gethead_NS30(EEG)) return CNTERR_FILE;
  if (EEG->eep_header.chanc < 1 || EEG->eep_header.period < CNT_MIN_PERIOD) return CNTERR_DATA;

  /* trigger table */
  if ((r = trg_read_NS30(EEG))) return r;

  /* multiplex buffer setup */
  EEG->store[DATATYPE_EEG].data.buf_int = (sraw_t *)
    v_malloc((size_t) EEG->store[DATATYPE_EEG].epochs.epochl * EEG->eep_header.chanc * sizeof(sraw_t), "buf");

  /* fill first buffer */
  EEG->store[DATATYPE_EEG].data.bufepoch = -2;
  return getepoch_NS30(EEG, 0);
}

int cntopen_AVR(eeg_t *EEG)
{
  EEG->mode = CNT_AVR;
  EEG->trg = trg_init();

  return gethead_AVR(EEG);
}

int cntopen_EEP20(eeg_t *EEG)
{
  FILE *f = EEG->f;
  uint64_t fsize;

  EEG->mode = CNT_EEP20;

  EEG->trg = trg_init();
  /* trigger table is initialized from per-sample flags during eep_read_sraw() */

  if (    eepio_fseek(f, 0, SEEK_END)
      || (fsize = eepio_ftell(f)) < 0
      || eepio_fseek(f, 0, SEEK_SET)
      || gethead_EEP20(EEG)       )
    return CNTERR_FILE;

  if (EEG->eep_header.chanc < 1 || EEG->eep_header.period < CNT_MIN_PERIOD)
    return CNTERR_DATA;

  EEG->eep_header.samplec =  (long) (fsize - SAMPLESTART_EEP20(EEG->eep_header.chanc))
                / SAMPLESIZE_EEP20(EEG->eep_header.chanc);
  if (  (fsize - SAMPLESTART_EEP20(EEG->eep_header.chanc))
      % SAMPLESIZE_EEP20(EEG->eep_header.chanc) )
    return CNTERR_DATA;

  if (eepio_fseek(f, SAMPLESTART_EEP20(EEG->eep_header.chanc), SEEK_SET))
    return CNTERR_FILE;

  return CNTERR_NONE;
}

int saveold_EEP20(eeg_t *dst, eeg_t *src, unsigned long delmask)
{
  char *buf; size_t n;
  int s, forget, child = 0;
  chunk_t srcchunk;
  fourcc_t id, listid;

  switch (src->mode) {
    case CNT_EEP20:
    case CNT_NS30:
      /* only ns header copy supported in EEP20 src */
      if (!(delmask & FORGET_nsh)) {
        n = SAMPLESTART_EEP20(dst->eep_header.chanc);
        buf = (char *) malloc(n);
        if (buf == NULL) return CNTERR_MEM;
        eepio_fseek(src->f, 0, SEEK_SET);
        eepio_fread(buf, n, 1, src->f);
        eepio_fseek(dst->f, 0, SEEK_SET);
        eepio_fwrite(buf, n, 1, dst->f);
        free(buf);
        if (ferror(src->f) || ferror(dst->f)) return CNTERR_FILE;
      }
      return CNTERR_NONE;

    case CNT_RIFF:
      // TODO check for 64-bit riff version
      while (!(s = riff_fetch(src->f, &srcchunk, &listid, src->cnt, child))) {

        /* forget all except of nsh if not masked */
        id = riff_get_chunk_id(srcchunk);
        forget = (id != FOURCC_nsh || delmask & FORGET_nsh);

        if (!forget) {
          n = (size_t) srcchunk.size;
          buf = (char *) malloc(n);
          if (buf == NULL) return CNTERR_MEM;
          riff_read(buf, n, 1, src->f, srcchunk);
          eepio_fseek(dst->f, 0, SEEK_SET);
          eepio_fwrite(buf, n, 1, dst->f);
          free(buf);
          if (ferror(src->f) || ferror(dst->f)) return CNTERR_FILE;
        }
        child++;
      }

      if (s != RIFFERR_NOCHUNK)
        return CNTERR_FILE;
      else
        return CNTERR_NONE;

    case CNT_AVR:
      return CNTERR_NONE;

    default:
      return CNTERR_DATA;
  }
}

int saveold_RAW3(eeg_t *dst, eeg_t *src, unsigned long delmask)
{
  char *buf;
  size_t n;
  int blockc, block;
  uint64_t size;
  chunk_t curchunk, srcchunk, dstchunk;
  fourcc_t id, listid;
  int s, forget, child = 0;

  switch (src->mode) {
    case CNT_EEP20:
    case CNT_NS30:
      /* only ns header copy supported in EEP20/NS30 src */
      if (!(delmask & FORGET_nsh)) {
        riff_new(dst->f, &curchunk, FOURCC_nsh, &dst->cnt);
        n = SAMPLESTART_EEP20(dst->eep_header.chanc);
        buf = (char *) malloc(n);
        if (buf == NULL) return CNTERR_MEM;
        eepio_fseek(src->f, 0, SEEK_SET);
        eepio_fread(buf, n, 1, src->f);
        riff_write(buf, n, 1, dst->f, &curchunk);
        free(buf);
        riff_close(dst->f, curchunk);
        if (ferror(src->f) || ferror(dst->f)) return CNTERR_FILE;
      }
      return CNTERR_NONE;

    /* raw3 src? - copy all (unknown) chunks which are not in delmask */
    case CNT_RIFF:
    case CNTX_RIFF:
      // TODO check for 64-bit riff version
      while (!(s = riff_fetch(src->f, &srcchunk, &listid, src->cnt, child))) {

        /* found a chunk, forget it ? */
        id = riff_get_chunk_id(srcchunk);
        forget = ((id == FOURCC_info) && (delmask & FORGET_info))
              || ((id == FOURCC_nsh)  && (delmask & FORGET_nsh))
              || ((id == FOURCC_evt)  && (delmask & FORGET_evt))
              || ((id == FOURCC_eeph) && (delmask & FORGET_eeph))
              || ((id == FOURCC_tfh)  && (delmask & FORGET_tfh))
              || ((id == FOURCC_LIST && listid == FOURCC_raw3)
                                      && (delmask & FORGET_raw))
              || (( id == FOURCC_LIST && listid == FOURCC_rawf)
                                      && (delmask & FORGET_rawf))
              || (( id == FOURCC_LIST && listid == FOURCC_stdd)
                                      && (delmask & FORGET_stdd))
              || (( id == FOURCC_LIST && listid == FOURCC_tfd)
                                      && (delmask & FORGET_tfd))
              || (( id == FOURCC_LIST && listid == FOURCC_imp)
                                      && (delmask & FORGET_imp));

        if (!forget) {
          size = srcchunk.size;
          if (id == FOURCC_LIST) {
            riff_list_new(dst->f, &dstchunk, listid, &dst->cnt);
            size -= 4;          /* 4 bytes list id */
            dst->cnt.size -= 4; /* ??? */
          }
          else {
            riff_new(dst->f, &dstchunk, id, &dst->cnt);
          }

          /* copy source chunk blockwise */
          buf = (char *) malloc(BUFSIZ);
          if (buf == NULL) return CNTERR_MEM;
          blockc = size / BUFSIZ + (size % BUFSIZ > 0);

          for (block = 0; block < blockc; block++) {
            n = BUFSIZ;
            if (block == blockc - 1)
              n = size % BUFSIZ;

            if (    riff_read(buf, n, 1, src->f, srcchunk)
                || riff_write(buf, n, 1, dst->f, &dstchunk)  )
            {
              free(buf);
              return CNTERR_FILE;
            }
          }
          free(buf);
          if (riff_close(dst->f, dstchunk))
            return CNTERR_FILE;
        }
        child++;
      }

      if (s != RIFFERR_NOCHUNK)
        return CNTERR_FILE;
      else
        return CNTERR_NONE;

    case CNT_AVR:
      return CNTERR_NONE;

    default:
      return CNTERR_DATA;
  }
}


int eep_create_file_EEP20(eeg_t *dst, eeg_t *src, unsigned long delmask)
{
  int s;

  if (src != NULL)
    if ((s = saveold_EEP20(dst, src, delmask))) return s;

  return eep_seek(dst, DATATYPE_EEG, 0, 0);
}

/** debugging stuff */
/* we do not want this prototype in raw3.h - therefore defined here */

int eep_write_sraw_EEP20 (eeg_t *cnt, sraw_t *muxbuf, sraw_t *statusflags, slen_t n)
{
  FILE *f = cnt->f;
  long step = cnt->eep_header.chanc;
  // size_t outbytes = (size_t) step * sizeof(sraw_t);
  slen_t i;

  if (CNT_EEP20 != cnt->mode)
    return CNTERR_BADREQ;

  for (i = 0; i < n; i++)
  {
    /* clear all trigger bits in the per-sample flag area
      (all valid triggers are in the EEG->trg list which is
      written completely in eep_finish_file) */
    statusflags[2 * i] &= ~EEP20_TRGMASK;

    if (vwrite_s16(f, &muxbuf[i*step], step) != step)
      return CNTERR_FILE;
    if (vwrite_s16(f, &statusflags[i*2], 2) != 2)
      return CNTERR_FILE;
  }

  return CNTERR_NONE;
}

void eep_set_keep_file_consistent(eeg_t *cnt, int enable)
{
  cnt->keep_consistent = enable;
}

val_t* eep_get_values(eeg_t* cnt)
{
  if(cnt->values)
    return cnt->values;

  /* It is possible to add extra information here when wanted. */
  val_create(&cnt->values);
  return cnt->values;
}

