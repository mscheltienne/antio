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

#ifndef CNT_H
#define CNT_H

#if defined(WIN32)
#include <TCHAR.H>
#else
#define TCHAR char
#endif

#define RCS_CNT_H "$RCSfile: cnt.h,v $ $Revision: 2415 $"

#include <time.h>
#include <eep/stdint.h>
#include <eep/eepmisc.h>
#include <cnt/trg.h>
#include <eep/val.h>

/*
  tags for supported "cnt" formats
*/
#define CNT_NS30   1
#define CNT_EEP20  2
/*#define CNT_RAW3   3*/
#define CNT_AVR    4

#define CNT_RIFF   5 /* General RIFF data type */
/* #define CNT_TF     5 */ /* Time/frequency mode (*no* EEG (RAW3) data in file!) */
/* #define CNT_TFRAW3 6 */ /* Both RAW3 + Time Frequency mode (e.g. both EEG + FFT/wavelet data) */
/* #define CNT_RAWF   7 */ /* AVR files saved in RIFF format just like CNT files (new style AVR's) */
#define CNTX_RIFF  8 /* 64-bit RIFF data type */


/*
  we have to export some EEP 2.0 format internals due to
  historical reasons
*/
#define GENHEADER_SIZE        900
#define CHANHEADER_SIZE  75
#define SAMPLESIZE_EEP20(chanc) (((chanc) + 2) * 2)
#define SAMPLESTART_EEP20(chanc) ((chanc) * CHANHEADER_SIZE + GENHEADER_SIZE)

#define EEP20_REJECT   0x0100
#define EEP20_REJMASK  0x0100


/* cnt-function status-values */
#define  CNTERR_NONE   0        /* no error */
#define  CNTERR_FILE   1        /* file access errors */
#define  CNTERR_LOCK   2        /* file locked by another process - unused since 3.0*/
#define  CNTERR_MEM    3        /* memory allocation errors */
#define  CNTERR_DATA   4        /* inconsistent / illegal contents in the file */
#define  CNTERR_RANGE  5        /* You asked for a sample that is out of range */
#define  CNTERR_BADREQ 6        /* Wrong program logic - you're asking something inherently impossible,
                                   such as reading data from an integer chunk using a float function */

#define FOURCC_CNT  FOURCC('C', 'N', 'T', ' ')
#define FOURCC_nsh  FOURCC('n', 's', 'h', ' ')

typedef enum
{
  CONTENT_UNKNOWN,
  COMPLEX_REAL,
  COMPLEX_IMAGINARY,
  COMPLEX_ABSOLUTE,
  COMPLEX_PHASE,
  CONTENT_POWER
} tf_content_e;

typedef enum
{
  DATATYPE_UNDEFINED = -1,
  DATATYPE_EEG = 0,       /* sraw */
  DATATYPE_TIMEFREQ = 1,  /* float  */
  DATATYPE_AVERAGE = 2,   /* float */
  DATATYPE_STDDEV = 3     /* float */
  /* If you add one, don't forget to increase NUM_DATATYPES below! */
} eep_datatype_e;

#define NUM_DATATYPES 4

struct record_info_s {
  double        m_startDate;      /* Segment Start Time, m_Date */
  double        m_startFraction;  /* Segment Start Time, m_Fraction */
  asciiline_t   m_szHospital;     /* Hospital ID or desciption */
  asciiline_t   m_szTestName;     /* Name of test */
  asciiline_t   m_szTestSerial;   /* Number of test */
  asciiline_t   m_szPhysician;    /* Name of physician */
  asciiline_t   m_szTechnician;   /* Name of technician */
  asciiline_t   m_szMachineMake;  /* Make of recording machine */
  asciiline_t   m_szMachineModel; /* Model of recording machine */
  asciiline_t   m_szMachineSN;    /* Recording machine serial # or ID */
  asciiline_t   m_szName;         /* Name of patient */
  asciiline_t   m_szID;           /* Patient ID */
  asciiline_t   m_szAddress;      /* Patient home address */
  asciiline_t   m_szPhone;        /* Patient phone number */
  asciiline_t   m_szComment;      /* Comment */
  TCHAR         m_chSex;          /* Capital letters M and F for male and female respectively */
  TCHAR         m_chHandedness;   /* Capital letters M, L or R for mixed, left or right */
  struct tm     m_DOB;            /* Date of birth */
};

typedef struct eegchan_s eegchan_t;
typedef struct tf_component_s tf_component_t;
typedef struct eeg_dummy_t eeg_t;
typedef struct record_info_s record_info_t;

/*
  converters need to create and fill a channel table
  in preparation to eep_init_from_values()
*/
eegchan_t *eep_chan_init(short chanc);
void       eep_chan_set (eegchan_t *chanv, short chan, const char *lab, double iscale, double rscale, const char *runit);
void       eep_chan_set_reflab(eegchan_t *chanv, short chan, const char *reflab);

void eep_free(eeg_t *cnt);

/*
  init a EEG access structure by parameters;
  you cannot eep_read_sraw structures initialized this way; it serves for
  creating data files "from scratch"
  the chanv space is not copied; you must NOT free it
*/
eeg_t *eep_init_from_values(double period, short chanc, eegchan_t *chanv);

/*
  register a opened readable stream (source .cnt file) to the EEG access
  structure which is initialized by reading the file

  caller must NOT use this stream after registering; eep_finish_file first!
  filename is copied to own space in eeg_t
  returns NULL and error code in status if failed
*/
eeg_t *eep_init_from_file(const char *fname, FILE *f, int *status);

/*
  init a destination EEG access structure from source
  (nothing happens with files, memory only)
  a subsequent eep_create_file will create a dest cnt file with the file mode
  and channel layout from the source
*/
eeg_t *eep_init_from_copy(eeg_t *src);


/*
  eep_create_file:
  register a opened writable stream (dest .cnt) to the EEG access structure
  caller must NOT use this stream after registering; eep_finish_file first!
  filename is copied to own space in eeg_t

  if src is not NULL on call, delmask controls
  which parts of source are NOT copied to destination file (either the
  calling application maintains the data itself or knows that data become invalid
  and want to forget it)

  the copy procedere works well on RIFF files only; in EEP20 it handles
  the NeuroScan / EEP 2.0 header only

  the registry string is added(copied) to the cnt history (RIFF files only)
*/

#define FORGET_none 0x00000000   /* copy all things found (never used I think) */
#define FORGET_raw  0x00000001   /* you have to forget this always !!! */
#define FORGET_eeph 0x00000002   /* you have to forget this always !!! */
#define FORGET_evt  0x00000004   /* you have to forget this always !!! */
#define FORGET_nsh  0x00000008

#define FORGET_std  (FORGET_raw | FORGET_eeph | FORGET_evt) /* this will fit for most apps */
#define FORGET_tfd  0x00000010
#define FORGET_tfh  0x00000020
#define FORGET_info 0x00000040
#define FORGET_rawf 0x00000080
#define FORGET_stdd 0x00000100
#define FORGET_imp  0x00000200

int eep_create_file(eeg_t *dst, const char *fname, FILE *f, eeg_t *src, unsigned long delmask, const char *registry);
int eep_create_file64(eeg_t *dst, const char *fname, FILE *f, const char *registry);


/*
  if writable, flush eeg_t memory buffers to output, append the access information tables
  free any memory allocated by the structure
  the assigned FILE* is NOT closed; it belongs to caller after this
  (seek undefined)
  return: status
*/
int eep_finish_file(eeg_t *cnt);

/*
  This does the same as eep_finish_file, except it will also fclose the filepointer!
  This is specificly meant for use when an eep file is closed at one point and closed at
  anothor.

  return: status
*/
int eep_fclose(eeg_t *cnt);

/*
  random cnt read access to each sample in record
  eep_write_sraw has to be sequential; no seeks allowed for output cnt's
  eep_write_sraw changes muxbuf contents!

  caller is responsible for buffer space - use CNTBUF_SIZE to calc the
  required space in bytes; buffer is channel multiplexed
  (EEP 2.0 control words - 2 additional 'channels' - not available)

  caller must prevent seeks/reads out of cnt scope
  return: status
*/

#define CNTBUF_SIZE(cnt, n) ((eep_get_chanc(cnt) ) * (n) * sizeof(sraw_t))
#define CNTBUF_ARRAYSIZE(cnt, n) ((eep_get_chanc(cnt) ) * (n))
#define TF_CNTBUF_SIZE(cnt, n) (eep_get_chanc(cnt) * eep_get_compc(cnt) * (n) * sizeof(float))
#define TF_CNTBUF_ARRAYSIZE(cnt, n) (eep_get_chanc(cnt) * eep_get_compc(cnt) * (n))
#define FLOAT_CNTBUF_SIZE(cnt, n) (eep_get_chanc(cnt) * (n) * sizeof(float))
#define FLOAT_CNTBUF_ARRAYSIZE(cnt, n) (eep_get_chanc(cnt) * (n))

int eep_seek   (eeg_t *cnt, eep_datatype_e type, uint64_t sample, int relative);
int eep_read_sraw   (eeg_t *cnt, eep_datatype_e type, sraw_t *muxbuf, uint64_t n);
int eep_read_float  (eeg_t *cnt, eep_datatype_e type, float  *muxbuf, uint64_t n);
/* For writing, the datatype depends on what has been set by eep_prepare_to_write(some_datatype) */
int eep_write_sraw  (eeg_t *cnt, const sraw_t *muxbuf, uint64_t n);
int eep_write_float (eeg_t *cnt, float  *muxbuf, uint64_t n);

/*
  return or set the cnt trigger archive handle
  set will free the old table memory
*/
trg_t *eep_get_trg(eeg_t *cnt);
void  eep_set_trg(eeg_t *cnt, trg_t *trg);

/*
  application access to eeg_t members
  set_... is only useful in output cnt's
  it's fatal to apply it to input!!!
*/
char *         eep_get_name(eeg_t *cnt);

short          eep_get_rate(eeg_t *cnt);
double         eep_get_period(eeg_t *cnt);
void           eep_set_period(eeg_t *cnt, double period);

short          eep_get_chanc(eeg_t *cnt);
void           eep_dup_chan(eeg_t *cnt, short chan, char *newlab);
uint64_t       eep_get_samplec(eeg_t *cnt);
int            eep_get_samplec_full(const eeg_t *cnt, uint64_t *samplec);

int            eep_get_chan_index(eeg_t *cnt, const char *lab);  /* case insens., -1 if not found */

char *         eep_get_chan_label(eeg_t *cnt,  short chan);
void           eep_set_chan_label(eeg_t *cnt,  short chan, const char *lab);

double         eep_get_chan_scale(eeg_t *cnt,  short chan);
void           eep_set_chan_scale(eeg_t *cnt,  short chan, double scale);

char *         eep_get_chan_unit(eeg_t *cnt,  short chan);
void           eep_set_chan_unit(eeg_t *cnt,  short chan, const char* unit);

double         eep_get_chan_iscale(eeg_t *cnt,  short chan);
void           eep_set_chan_iscale(eeg_t *cnt,  short chan, double scale);

double         eep_get_chan_rscale(eeg_t *cnt,  short chan);
void           eep_set_chan_rscale(eeg_t *cnt,  short chan, double scale);

void           eep_set_chan_reflab(eeg_t *cnt, short chan, const char *reflab);
char*          eep_get_chan_reflab(eeg_t *cnt, short chan);

void           eep_set_chan_status(eeg_t *cnt, short chan, const char *status);
char*          eep_get_chan_status(eeg_t *cnt, short chan);

void           eep_set_chan_type(eeg_t *cnt, short chan, const char *status);
char*          eep_get_chan_type(eeg_t *cnt, short chan);

/*
  get input and force output file mode; only converters need to call this

  a switch to CNT_RAW3 mode requires compression parameters,
  chanc has to be initialized before this call, length of chanv is chanc
  if chanv is NULL, the native channel order is used
*/
int            eep_get_mode(eeg_t *cnt);
void           eep_set_mode_EEP20(eeg_t *cnt);

int            eep_prepare_to_write(eeg_t *cnt, eep_datatype_e type, uint64_t epochl, short *chanv);

/********************* Added functions for handling Time/Frequency data **********/

void eep_comp_set(tf_component_t *compv, short comp, float value, const char *descr);
tf_component_t* eep_comp_init(short compc);
eeg_t* eep_init_from_tf_values(double period, short chanc, eegchan_t *chanv, short compc, tf_component_t *compv);
int eep_get_compc(eeg_t* cnt);
int eep_dup_comp(eeg_t *cnt, short comp, float newvalue);
int eep_get_comp_index(eeg_t *cnt, float value);
float eep_get_comp_value(eeg_t *cnt, int comp_index);
int eep_set_comp_value(eeg_t *cnt, int comp_index, float value);
const char *eep_get_comp_description(eeg_t *cnt, int comp_index);
int eep_set_comp_description(eeg_t *cnt, int comp_index, const char *description);
const char *eep_get_comp_axis_unit(eeg_t *cnt);
void eep_set_comp_axis_unit(eeg_t *cnt, const char* unit);
const char* eep_get_tf_type(eeg_t *cnt);
void eep_set_tf_type(eeg_t *cnt, const char *tf_type);
tf_content_e eep_get_tf_contenttype(eeg_t *cnt);
void eep_set_tf_contenttype(eeg_t *cnt, tf_content_e tf_contenttype);

double eep_get_tf_period(eeg_t *cnt);
void eep_set_tf_period(eeg_t *cnt, double period);
short eep_get_tf_rate(eeg_t *cnt);
uint64_t eep_get_tf_samplec(eeg_t *cnt);

const char* eep_get_tf_chan_unit(eeg_t *cnt, int chan_index);


/* Recording info */
/* The structure will be internally copied in the library, so you can keep
   your own record_info structures on the stack if you want.
*/

int  eep_has_recording_info(eeg_t* cnt);
void eep_set_recording_info(eeg_t *cnt, record_info_t* info);
/* Will copy the internal recording_info to the struct pointed to by 'info' */
void eep_get_recording_info(eeg_t *cnt, record_info_t* info);

long eep_get_total_trials(eeg_t *cnt);
long eep_get_averaged_trials(eeg_t *cnt);
const char* eep_get_conditionlabel(eeg_t *cnt);
const char* eep_get_conditioncolor(eeg_t *cnt);
double eep_get_pre_stimulus_interval(eeg_t *cnt);

void eep_set_total_trials(eeg_t *cnt, long total_trials);
void eep_set_averaged_trials(eeg_t *cnt, long averaged_trials);
void eep_set_conditionlabel(eeg_t *cnt, const char* conditionlabel);
void eep_set_conditioncolor(eeg_t *cnt, const char* conditioncolor);
void eep_set_pre_stimulus_interval(eeg_t *cnt, double pre_stimulus);
/* convert to and fro excel date */
void eep_exceldate_to_unixdate(double, double, time_t *);
void eep_unixdate_to_exceldate(time_t, double *, double *);
/* general purpose time reference functions in various incarnations */
time_t eep_get_time_epoch(double time, double fraction);
void eep_get_time_string(double time, double fraction, char *s);
void eep_get_time_struct(double time, double fraction, struct tm *);
/* get set the recording startdate in various incarnations */
time_t eep_get_recording_startdate_epoch(eeg_t *);
void eep_get_recording_startdate_string(eeg_t *, char *);
void eep_get_recording_startdate_struct(eeg_t *, struct tm *);
void eep_set_recording_startdate_epoch(eeg_t *, time_t);
void eep_set_recording_startdate_struct(eeg_t *, struct tm *);

const char *eep_get_hospital(eeg_t *cnt);
const char *eep_get_test_name(eeg_t *cnt);
const char *eep_get_test_serial(eeg_t *cnt);
const char *eep_get_physician(eeg_t *cnt);
const char *eep_get_technician(eeg_t *cnt);
const char *eep_get_machine_make(eeg_t *cnt);
const char *eep_get_machine_model(eeg_t *cnt);
const char *eep_get_machine_serial_number(eeg_t *cnt);
const char *eep_get_patient_name(eeg_t *cnt);
const char *eep_get_patient_id(eeg_t *cnt);
const char *eep_get_patient_address(eeg_t *cnt);
const char *eep_get_patient_phone(eeg_t *cnt);
char eep_get_patient_sex(eeg_t *cnt);
char eep_get_patient_handedness(eeg_t *cnt);
struct tm *eep_get_patient_day_of_birth(eeg_t *cnt);
const char *eep_get_comment(eeg_t *cnt);

void eep_set_hospital(eeg_t *cnt, const char *value);
void eep_set_test_name(eeg_t *cnt, const char *value);
void eep_set_test_serial(eeg_t *cnt, const char *value);
void eep_set_physician(eeg_t *cnt, const char *value);
void eep_set_technician(eeg_t *cnt, const char *value);
void eep_set_machine_make(eeg_t *cnt, const char *value);
void eep_set_machine_model(eeg_t *cnt, const char *value);
void eep_set_machine_serial_number(eeg_t *cnt, const char *value);
void eep_set_patient_name(eeg_t *cnt, const char *value);
void eep_set_patient_id(eeg_t *cnt, const char *value);
void eep_set_patient_address(eeg_t *cnt, const char *value);
void eep_set_patient_phone(eeg_t *cnt, const char *value);
void eep_set_patient_sex(eeg_t *cnt, char value);
void eep_set_patient_handedness(eeg_t *cnt, char value);
void eep_set_patient_day_of_birth(eeg_t *cnt, struct tm *value);
void eep_set_comment(eeg_t *cnt, const char *value);


/* Enable this if you want to have a valid output file at all times
   during write. Enable this when recording EEG's over very long intervals
   If the application crashes or power fails, the resulting file should still
   be usable. Default = off */
void           eep_set_keep_file_consistent(eeg_t *cnt, int enable);

int            eep_has_data_of_type(eeg_t *cnt, eep_datatype_e type);
int            eep_get_epochl(eeg_t *cnt, eep_datatype_e type);
short*         eep_get_chanseq(eeg_t *cnt, eep_datatype_e type);
void           eep_get_fileversion(eeg_t *cnt, char *version);
int            eep_get_fileversion_major(eeg_t *cnt);
int            eep_get_fileversion_minor(eeg_t *cnt);

/* eep_get_data_format return values (string in format buffer)
    "NeuroScan 3.x (16 bit blocked)"
    "NeuroScan 4.1 (16 bit channel multiplexed)"
    "EEP 2.0 (16 bit channel multiplexed)"
    "EEP 3.x (32 bit raw3 compressed)"
    "EEP <version_major>.<version_minor> \
        time-frequency average stddev (float vectors)"
    "EEP 2.0/3.x avr (float vectors)"
    "unknown" */
void           eep_get_dataformat(eeg_t *cnt, char *format);

int            eep_has_history(eeg_t *);
void           eep_set_history(eeg_t *cnt, const char *hist);
void           eep_append_history(eeg_t *cnt, const char *histline);
const char*    eep_get_history(eeg_t *cnt);

int            eep_get_neuroscan_type(eeg_t *cnt); /* Backwards compat. */
int eep_write_sraw_EEP20 (eeg_t *cnt, sraw_t *muxbuf, sraw_t *statusflags, slen_t n); /* Backwards compat. */

val_t*         eep_get_values(eeg_t* cnt);
#endif
