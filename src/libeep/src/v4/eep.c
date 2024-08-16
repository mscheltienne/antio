// system
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <string.h>
#include <time.h>
// libeep
#include <v4/eep.h>
#include <cnt/evt.h>
#include <cnt/seg.h>
#include <cnt/cnt.h>
#include <cnt/trg.h>
#include <eep/eepio.h> // for the definition of eepio_fopen
#include <cnt/cnt_private.h> // for the definition of eegchan_s
///////////////////////////////////////////////////////////////////////////////
#define SCALING_FACTOR 128
///////////////////////////////////////////////////////////////////////////////
typedef enum { dt_none, dt_avr, dt_cnt } data_type;
typedef enum { om_none, om_read, om_write } open_mode;
///////////////////////////////////////////////////////////////////////////////
struct _libeep_trigger_extension_mutable {
  int32_t    type;
  int32_t    code;
  uint64_t   duration_in_samples;
  char     * condition;
  char     * description;
  char     * videofilename;
  char     * impedances;
};
struct _processed_trigger {
  char                                     * label;
  uint64_t                                   sample;
  struct _libeep_trigger_extension_mutable   te;
};
///////////////////////////////////////////////////////////////////////////////
struct _libeep_entry {
  FILE      * file;
  eeg_t     * eep;
  data_type   data_type;
  open_mode   open_mode;
  float     * scales;
  // processed trigger data
  int                         processed_trigger_count;
  struct _processed_trigger * processed_trigger_data;
};

struct _libeep_channels {
  eegchan_t *channels;
  short count;
};

static struct _libeep_entry ** _libeep_entry_map;
static struct record_info_s ** _libeep_recinfo_map;
static struct _libeep_channels ** _libeep_channel_map;

static int _libeep_entry_size;
static int _libeep_recinfo_size;
static int _libeep_channel_size;
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_entry_map and _libeep_entry_size */
static cntfile_t
_libeep_allocate() {
  struct _libeep_entry **new_entry_map = NULL;
  new_entry_map = realloc(_libeep_entry_map, sizeof(struct _libeep_entry *) * (_libeep_entry_size + 1));
  if (new_entry_map == NULL) {
    return -1;
  }
  _libeep_entry_map = new_entry_map;
  _libeep_entry_map[_libeep_entry_size]=(struct _libeep_entry *)malloc(sizeof(struct _libeep_entry));
  if (_libeep_entry_map[_libeep_entry_size] == NULL) {
    return -1;
  }
  _libeep_entry_map[_libeep_entry_size]->open_mode=om_none;
  _libeep_entry_map[_libeep_entry_size]->data_type=dt_none;
  _libeep_entry_size += 1;
  return _libeep_entry_size - 1;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_entry_map and _libeep_entry_size */
static void
_libeep_free(cntfile_t handle) {
  if(_libeep_entry_map[handle]==NULL) {
    fprintf(stderr, "libeep: cannot free cnt handle %i\n", handle);
    return;
  }
  // close handle
  free(_libeep_entry_map[handle]);
  // set null
  _libeep_entry_map[handle]=NULL;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_entry_map and _libeep_entry_size */
static void
_libeep_free_map() {
  int i;
  for (i = 0; i < _libeep_entry_size; ++i) {
    if (_libeep_entry_map[i] != NULL) {
      _libeep_free(i); // TODO: or use libeep_close?
    }
  }
  if (_libeep_entry_map != NULL) {
    free(_libeep_entry_map);
  }
  _libeep_entry_map = NULL;
  _libeep_entry_size = 0;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_entry_map and _libeep_entry_size */
static struct _libeep_entry *
_libeep_get_object(cntfile_t handle, open_mode om) {
  struct _libeep_entry *rv = NULL;
  if (handle < 0) {
    fprintf(stderr, "libeep: invalid cnt handle %i\n", handle);
    exit(-1);
  }
  if (_libeep_entry_map==NULL) {
    fprintf(stderr, "libeep: cnt entry map not initialized\n");
    exit(-1);
  }
  if (handle >= _libeep_entry_size) {
    fprintf(stderr, "libeep: invalid cnt handle %i\n", handle);
    exit(-1);
  }
  rv = _libeep_entry_map[handle];
  // check valid handle
  if(rv==NULL) {
    fprintf(stderr, "libeep: invalid cnt handle %i\n", handle);
    exit(-1);
  }
  // check valid open mode
  if(om != om_none && rv->open_mode != om) {
    fprintf(stderr, "libeep: invalid mode on cnt handle %i\n", handle);
    exit(-1);
  }

  return rv;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_recinfo_map and _libeep_recinfo_size */
static recinfo_t
_libeep_recinfo_allocate() {
  struct record_info_s ** new_recinfo_map = NULL;
  new_recinfo_map = realloc(_libeep_recinfo_map, sizeof(struct record_info_s *) * (_libeep_recinfo_size + 1));
  if (new_recinfo_map == NULL) {
    return -1;
  }
  _libeep_recinfo_map = new_recinfo_map;
  _libeep_recinfo_map[_libeep_recinfo_size] = (struct record_info_s *)malloc(sizeof(struct record_info_s));
  if (_libeep_recinfo_map[_libeep_recinfo_size] == NULL) {
    return -1;
  }
  memset(_libeep_recinfo_map[_libeep_recinfo_size], 0, sizeof(struct record_info_s));
  // set default values to prevent recording info line corruption
  _libeep_recinfo_map[_libeep_recinfo_size]->m_chSex = ' ';
  _libeep_recinfo_map[_libeep_recinfo_size]->m_chHandedness = ' ';
  _libeep_recinfo_size += 1;
  return _libeep_recinfo_size - 1;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_recinfo_map and _libeep_recinfo_size */
static void
_libeep_recinfo_free(recinfo_t handle) {
  if (_libeep_recinfo_map[handle] == NULL) {
    fprintf(stderr, "libeep: cannot free recording info handle %i\n", handle);
    return;
  }
  // close handle
  free(_libeep_recinfo_map[handle]);
  // set null
  _libeep_recinfo_map[handle] = NULL;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_recinfo_map and _libeep_recinfo_size */
static struct record_info_s *
_libeep_get_recinfo(recinfo_t handle) {
  struct record_info_s *rv = NULL;
  if (handle < 0) {
    fprintf(stderr, "libeep: invalid recording info handle %i\n", handle);
    exit(-1);
  }
  if (_libeep_recinfo_map == NULL) {
    fprintf(stderr, "libeep: recording info map not initialized\n");
    exit(-1);
  }
  if (handle >= _libeep_recinfo_size) {
    fprintf(stderr, "libeep: invalid recording info handle %i\n", handle);
    exit(-1);
  }
  rv = _libeep_recinfo_map[handle];
  // check valid handle
  if (rv == NULL) {
    fprintf(stderr, "libeep: invalid recording info handle %i\n", handle);
    exit(-1);
  }
  return rv;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_recinfo_map and _libeep_recinfo_size */
static void
_libeep_free_recinfo_map() {
  int i;
  for (i = 0; i < _libeep_recinfo_size; ++i) {
    if (_libeep_recinfo_map[i] != NULL) {
      _libeep_recinfo_free(i);
    }
  }
  if (_libeep_recinfo_map != NULL) {
    free(_libeep_recinfo_map);
  }
  _libeep_recinfo_map = NULL;
  _libeep_recinfo_size = 0;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_channel_map and _libeep_channel_size */
static chaninfo_t
_libeep_channels_allocate() {
  struct _libeep_channels ** new_channel_map = NULL;
  new_channel_map = realloc(_libeep_channel_map, sizeof(struct _libeep_channels *) * (_libeep_channel_size + 1));
  if (new_channel_map == NULL) {
    return -1;
  }
  _libeep_channel_map = new_channel_map;
  _libeep_channel_map[_libeep_channel_size] = (struct _libeep_channels *)malloc(sizeof(struct _libeep_channels));
  if (_libeep_channel_map[_libeep_channel_size] == NULL) {
    return -1;
  }
  _libeep_channel_map[_libeep_channel_size]->channels = NULL;
  _libeep_channel_map[_libeep_channel_size]->count = 0;
  _libeep_channel_size += 1;
  return _libeep_channel_size - 1;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_channel_map and _libeep_channel_size */
static void
_libeep_channels_free(chaninfo_t handle) {
  if (_libeep_channel_map[handle] == NULL) {
    fprintf(stderr, "libeep: cannot free channel info handle %i\n", handle);
    return;
  }
  if (_libeep_channel_map[handle]->channels != NULL) {
    free(_libeep_channel_map[handle]->channels);
  }
  // close handle
  free(_libeep_channel_map[handle]);
  // set null
  _libeep_channel_map[handle] = NULL;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_channel_map and _libeep_channel_size */
static struct _libeep_channels *
_libeep_get_channels(chaninfo_t handle) {
  struct _libeep_channels *rv = NULL;
  if (handle < 0) {
    fprintf(stderr, "libeep: invalid channel info handle %i\n", handle);
    exit(-1);
  }
  if (_libeep_channel_map == NULL) {
    fprintf(stderr, "libeep: channel info map not initialized\n");
    exit(-1);
  }
  if (handle >= _libeep_channel_size) {
    fprintf(stderr, "libeep: invalid channel info handle %i\n", handle);
    exit(-1);
  }
  rv = _libeep_channel_map[handle];
  // check valid handle
  if (rv == NULL) {
    fprintf(stderr, "libeep: invalid channel info handle %i\n", handle);
    exit(-1);
  }
  return rv;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper for manipulating _libeep_channel_map and _libeep_channel_size */
static void
_libeep_free_channels_map() {
  int i;
  for (i = 0; i < _libeep_channel_size; ++i) {
    if (_libeep_channel_map[i] != NULL) {
      _libeep_channels_free(i);
    }
  }
  if (_libeep_channel_map != NULL) {
    free(_libeep_channel_map);
  }
  _libeep_channel_map = NULL;
  _libeep_channel_size = 0;
}
///////////////////////////////////////////////////////////////////////////////
/* local helper to return string with the end replaced */
static char *
_replace_string_end(const char * input, const char * end) {
  char   * input_dup;
  size_t   input_len;
  size_t   end_len;

  input_len = strlen(input);
  end_len = strlen(end);

  if(input_len > end_len) {
    input_dup = (char *)malloc(input_len + 1);
    strcpy(input_dup, input);
    sprintf(input_dup + input_len - end_len, end);
    return input_dup;
  }

  return NULL;
}
///////////////////////////////////////////////////////////////////////////////
/* convert libeep trg_t structure to processed trigger structure */
static void
_libeep_trg_t_to_processed(const trg_t * external_trg, struct _libeep_entry * obj) {
  int i;
  obj->processed_trigger_count = trg_get_c(external_trg);
  obj->processed_trigger_data = (struct _processed_trigger *)malloc(sizeof(struct _processed_trigger) * obj->processed_trigger_count);
  memset(obj->processed_trigger_data, 0, sizeof(struct _processed_trigger) * obj->processed_trigger_count);
  for(i=0;i<obj->processed_trigger_count;++i) {
    char *code;
    code=trg_get(external_trg, i, &obj->processed_trigger_data[i].sample);
  
    obj->processed_trigger_data[i].label = (char *)malloc(strlen(code) + 1);
    strcpy(obj->processed_trigger_data[i].label, code);
  }
}
///////////////////////////////////////////////////////////////////////////////
static double
_libeep_helper_excel_to_double(double excel, double fraction) {
  double return_value=0;

  // 27538 -> 1970jan1
  // 2958464 -> 30dec9999
  if(excel >= 27538 && excel <= 2958464) {
    return_value=((excel*3600.0*24.0) - 2209161600) + fraction;
  }

  return return_value;
}
///////////////////////////////////////////////////////////////////////////////
/* helper to convert excel date to offset
 */
static int
_libeep_helper_date_fraction_to_offset(double rate, double start_date, double start_fraction, double check_date, double check_fraction, uint64_t * offset) {
  double start_double = _libeep_helper_excel_to_double(start_date, start_fraction);
  double check_double = _libeep_helper_excel_to_double(check_date, check_fraction);
  double diff_double = (check_double - start_double);
/*
  fprintf(stderr, "to offset(%lf, %lf, %lf, %lf, %lf): ", rate, start_date, start_fraction, check_date, check_fraction);
  fprintf(stderr, "  cnt double: %lf: ", start_double);
  fprintf(stderr, "  evt double: %lf: ", check_double);
  fprintf(stderr, "  diff double: %lf: ", diff_double);
  fprintf(stderr, "  offset: %i\n", (int)(rate * diff_double));
*/

  *offset = rate * diff_double;

  return (check_date > 0 && check_date >= start_date && diff_double >= 0);
}
/* helper to check dates in segments
 */
static int
_libeep_helper_date_fraction_seg_to_offset(uint64_t sample_count, double rate, double start_date, double start_fraction, double check_date, double check_fraction, const libeep_seg_t * seg, uint64_t * offset) {
  // TODO
  if(seg) {
    int seg_index = seg->count;
    while(seg_index) {
      --seg_index;

      sample_count -= seg->array[seg_index].sample_count;
      if(_libeep_helper_date_fraction_to_offset(rate, seg->array[seg_index].date, seg->array[seg_index].fraction, check_date, check_fraction, offset)) {
        *offset += sample_count;
        return 1;
      }
    }
  }
  return _libeep_helper_date_fraction_to_offset(rate, start_date, start_fraction, check_date, check_fraction, offset);
}
///////////////////////////////////////////////////////////////////////////////
/* process triggers
 * try .trg file first
 * if not found, get the triggers from the cnt
 */
static void
_libeep_init_processed_triggers(const char * filename, struct _libeep_entry * obj, int external_triggers) {
  char * external_evt_filename;
  char * external_seg_filename;
  char * external_trg_filename;
  if(external_triggers) {
    // .evt file
    external_evt_filename = _replace_string_end(filename, "evt");
    external_seg_filename = _replace_string_end(filename, "seg");
    if(external_evt_filename) {
      // struct tm            cnt_time = {0};
      int                  pass;
      double               cnt_date = 0;
      double               cnt_fraction = 0;
      record_info_t        rec_inf;
      libeep_evt_event_t * e;
      libeep_evt_t       * evt = libeep_evt_read(external_evt_filename);
      libeep_seg_t       * seg = libeep_seg_read(external_seg_filename);
      free(external_evt_filename);
      free(external_seg_filename);

      // cnt time stamp
      if (eep_has_recording_info(obj->eep)) {
        eep_get_recording_info(obj->eep, &rec_inf);
        cnt_date = rec_inf.m_startDate;
        cnt_fraction = rec_inf.m_startFraction;
      }

      if(evt != NULL) {
        int i=0;
        obj->processed_trigger_count=0;
        // in pass 0, count triggers,
        // in pass 1, copy triggers
        for(pass=0;pass<2;++pass) {
          e=evt->evt_list_first;
          while(e != NULL) {
            uint64_t offset;
            if(_libeep_helper_date_fraction_seg_to_offset(eep_get_samplec(obj->eep), 1.0 / eep_get_period(obj->eep), cnt_date, cnt_fraction, e->time_stamp.date, e->time_stamp.fraction, seg, & offset)) {
              if(offset < eep_get_samplec(obj->eep)) {
                if(pass==0) {
                  obj->processed_trigger_count++;
                } else {
                  char tmp[16];
                  sprintf(tmp, "%i", e->code);
                  obj->processed_trigger_data[i].label = strdup(tmp);

                  obj->processed_trigger_data[i].sample = offset;
                  obj->processed_trigger_data[i].te.duration_in_samples = e->duration / eep_get_period(obj->eep);

                  obj->processed_trigger_data[i].te.type = e->type;
                  obj->processed_trigger_data[i].te.code = e->code;
                  if(e->condition) {
                    obj->processed_trigger_data[i].te.condition = strdup(e->condition);
                  }
                  if(e->description) {
                    obj->processed_trigger_data[i].te.description = strdup(e->description);
                  }
                  if(e->videofilename) {
                    obj->processed_trigger_data[i].te.videofilename = strdup(e->videofilename);
                  }
                  if(e->impedances) {
                    obj->processed_trigger_data[i].te.impedances = strdup(e->impedances);
                  }

                  ++i;
                }
              }
            }
            e = e->next_event;
          }
          if(pass==0 && obj->processed_trigger_count) {
            obj->processed_trigger_data = (struct _processed_trigger *)malloc(sizeof(struct _processed_trigger) * obj->processed_trigger_count);
            memset(obj->processed_trigger_data, 0, sizeof(struct _processed_trigger) * obj->processed_trigger_count);
          }
        }
        libeep_evt_delete(evt);
        libeep_seg_delete(seg);
      }
    }
    // return if triggers were found
    if(obj->processed_trigger_count) {
      return;
    }

    // .trg file
    external_trg_filename = _replace_string_end(filename, "trg");
    if(external_trg_filename) {
      FILE *external_trg_file=eepio_fopen(external_trg_filename, "r");
      if(external_trg_file) {
        trg_t * external_trg = trg_file_read(external_trg_file, eep_get_period(obj->eep));
        if(external_trg) {
          _libeep_trg_t_to_processed(external_trg, obj);

          trg_free(external_trg);
        }
        fclose(external_trg_file);
      }
      free(external_trg_filename);
    }
    // return if triggers were found
    if(obj->processed_trigger_count) {
      return;
    }
  }

  // embedded triggers
  _libeep_trg_t_to_processed(obj->eep->trg, obj);
  // return if triggers were found
  if(obj->processed_trigger_count) {
    return;
  }
#if 0
  // dummy data
  obj->processed_trigger_count = 1;
  obj->processed_trigger_data = (struct _processed_trigger *)malloc(sizeof(struct _processed_trigger) * obj->processed_trigger_count);
  obj->processed_trigger_data[0].sample = 13;
  const char * dummy_code = "dummy13";
  obj->processed_trigger_data[0].label = (char *)malloc(strlen(dummy_code) + 1);
  strcpy(obj->processed_trigger_data[0].label, dummy_code);
#endif
}
///////////////////////////////////////////////////////////////////////////////
/* clean up processed triggers resource */
static void
_libeep_fini_processed_triggers(struct _libeep_entry * obj) {
  if(obj->processed_trigger_data) {
    int i;
    for(i=0;i<obj->processed_trigger_count;++i) {
      free(obj->processed_trigger_data[i].label);
      free(obj->processed_trigger_data[i].te.condition);
      free(obj->processed_trigger_data[i].te.description);
      free(obj->processed_trigger_data[i].te.videofilename);
      free(obj->processed_trigger_data[i].te.impedances);
    }
    free(obj->processed_trigger_data);
    obj->processed_trigger_count = 0;
    obj->processed_trigger_data = NULL;
  }
}
///////////////////////////////////////////////////////////////////////////////
void libeep_init() {
  _libeep_entry_map = NULL;
  _libeep_entry_size = 0;
  _libeep_recinfo_map = NULL;
  _libeep_recinfo_size = 0;
  _libeep_channel_map = NULL;
  _libeep_channel_size = 0;
}
///////////////////////////////////////////////////////////////////////////////
void libeep_exit() {
  _libeep_free_map();
  _libeep_free_recinfo_map();
  _libeep_free_channels_map();
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_version() {
  static char version_string[128];
  snprintf(version_string, 128, "%i.%i.%i", LIBEEP_VERSION_MAJOR, LIBEEP_VERSION_MINOR, LIBEEP_VERSION_PATCH);
  return version_string;
}
///////////////////////////////////////////////////////////////////////////////
cntfile_t
_libeep_read_delegate(const char *filename, int external_triggers) {
  int status;
  int handle=_libeep_allocate();
  int channel_id;
  int channel_count;
  struct _libeep_entry * obj=_libeep_get_object(handle, om_none);
  // open file
  obj->file=eepio_fopen(filename, "rb");
  if(obj->file==NULL) {
    fprintf(stderr, "libeep: cannot open(1) %s\n", filename);
    _libeep_free(handle);
    return -1;
  }
  // eep struct
  obj->eep=eep_init_from_file(filename, obj->file, &status);
  if(status != CNTERR_NONE) {
    fprintf(stderr, "libeep: cannot open(2) %s\n", filename);
    return -1;
  }
  // read channel scale
  channel_count=eep_get_chanc(obj->eep);
  obj->scales=(float *)malloc(sizeof(float) * channel_count);
  for(channel_id=0; channel_id<channel_count; channel_id++) {
    obj->scales[channel_id]=(float)eep_get_chan_scale(obj->eep, channel_id);
  }
  // prepare structures for external trigger files
  obj->processed_trigger_count = 0;
  obj->processed_trigger_data = NULL;
  _libeep_init_processed_triggers(filename, obj, external_triggers);
  // housekeeping
  obj->open_mode=om_read;
  if(eep_has_data_of_type(obj->eep, DATATYPE_AVERAGE)) {
    obj->data_type=dt_avr;
  }
  if(eep_has_data_of_type(obj->eep, DATATYPE_EEG))     {
    obj->data_type=dt_cnt;
  }
  return handle;
}
///////////////////////////////////////////////////////////////////////////////
cntfile_t
libeep_read(const char *filename) {
  return _libeep_read_delegate(filename, 0);
}
///////////////////////////////////////////////////////////////////////////////
cntfile_t
libeep_read_with_external_triggers(const char *filename) {
  return _libeep_read_delegate(filename, 1);
}
///////////////////////////////////////////////////////////////////////////////
cntfile_t
libeep_write_cnt(const char *filename, int rate, chaninfo_t channel_info_handle, int rf64) {
  int cf;
  eegchan_t *channel_structure;
  int handle=_libeep_allocate();
  struct _libeep_entry * obj=_libeep_get_object(handle, om_none);
  struct _libeep_channels * channels_obj = _libeep_get_channels(channel_info_handle);
  // open file
  obj->file=eepio_fopen(filename, "wb");
  if(obj->file==NULL) {
    fprintf(stderr, "libeep: cannot open(1) %s\n", filename);
    return -1;
  }
  // channel setup
  channel_structure = eep_chan_init(channels_obj->count);
  if(channel_structure==NULL) {
    fprintf(stderr, "error in eep_chan_init!\n");
    return -1;
  }
  memmove(channel_structure, channels_obj->channels, sizeof(eegchan_t)* channels_obj->count);
  // file init
  obj->eep = eep_init_from_values(1.0 / (double)rate, channels_obj->count, channel_structure);
  if(obj->eep==NULL) {
    fprintf(stderr, "error in eep_init_from_values!\n");
    return -1;
  }
  // eep struct
  if(rf64) {
    cf=eep_create_file64(obj->eep, filename, obj->file, filename);
  } else {
    cf=eep_create_file(obj->eep, filename, obj->file, NULL, 0, filename);
  }
  if(cf != CNTERR_NONE) {
    fprintf(stderr, "could not create file!\n");
    return -1;
  }
  // switch writing mode
  if(eep_prepare_to_write(obj->eep, DATATYPE_EEG, rate, NULL) != CNTERR_NONE) {
    fprintf(stderr, "could not prepare file!\n");
    return -1;
  }
  eep_set_keep_file_consistent(obj->eep, 1);
  // scalings
  obj->scales = (float *)malloc(sizeof(float)* channels_obj->count);
  // prepare structures for external trigger files
  obj->processed_trigger_count = 0;
  obj->processed_trigger_data = NULL;
  // housekeeping
  obj->open_mode=om_write;
  obj->data_type=dt_cnt;
  return handle;
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_close(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_none);
  // close writing
  if(obj->open_mode==om_write) {
    eep_finish_file(obj->eep);
  }
  // close reading
  if(obj->open_mode==om_read) {
    eep_free(obj->eep);
  }
  // close scales
  free(_libeep_entry_map[handle]->scales);
  // clear structures for external trigger files
  _libeep_fini_processed_triggers(obj);
  // cleanup
  eepio_fclose(obj->file);
  _libeep_free(handle);
}
///////////////////////////////////////////////////////////////////////////////
int
libeep_get_channel_count(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chanc(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_channel_label(cntfile_t handle, int index) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chan_label(obj->eep, index);
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_channel_status(cntfile_t handle, int index) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chan_status(obj->eep, index);
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_channel_type(cntfile_t handle, int index) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chan_type(obj->eep, index);
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_channel_unit(cntfile_t handle, int index) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chan_unit(obj->eep, index);
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_channel_reference(cntfile_t handle, int index) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chan_reflab(obj->eep, index);
}
///////////////////////////////////////////////////////////////////////////////
float
libeep_get_channel_scale(cntfile_t handle, int index) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return (float)eep_get_chan_scale(obj->eep, index);
}
///////////////////////////////////////////////////////////////////////////////
int
libeep_get_channel_index(cntfile_t handle, const char *chan) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_chan_index(obj->eep, chan);
}
///////////////////////////////////////////////////////////////////////////////
int
libeep_get_sample_frequency(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return (int)(/* TODO: round before truncating */(1.0 / eep_get_period(obj->eep)));
}
///////////////////////////////////////////////////////////////////////////////
long
libeep_get_sample_count(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  return eep_get_samplec(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
static float *
_libeep_get_samples_avr(struct _libeep_entry * obj, long from, long to) {
  float *buffer_unscaled,
        *buffer_scaled;
  const float * ptr_src,
        * ptr_scales;
  float * ptr_dst;
  int n;
  int w;
  // seek
  if(eep_seek(obj->eep, DATATYPE_AVERAGE, from, 0)) {
    return NULL;
  }
  // get unscaled data
  buffer_unscaled = (float *)malloc(FLOAT_CNTBUF_SIZE(obj->eep, to-from));
  if(eep_read_float(obj->eep, DATATYPE_AVERAGE, buffer_unscaled, to-from)) {
    free(buffer_unscaled);
    return NULL;
  }
  // scale data
  buffer_scaled = (float *)malloc(sizeof(float) * (to-from) * eep_get_chanc(obj->eep));
  ptr_src=buffer_unscaled,
  ptr_scales=obj->scales;
  ptr_dst=buffer_scaled;
  n=eep_get_chanc(obj->eep) * (to-from);
  w = 0;
  while(n--) {
    if(!w) {
      w=to-from;
      ptr_scales=obj->scales;
    }
    *ptr_dst++ = (float)(*ptr_src++) **ptr_scales++;
    w--;
  }
  free(buffer_unscaled);
  return buffer_scaled;
  // TODO
}
///////////////////////////////////////////////////////////////////////////////
static float *
_libeep_get_samples_cnt(struct _libeep_entry * obj, long from, long to) {
  sraw_t *buffer_unscaled;
  float * buffer_scaled;
  const sraw_t * ptr_src;
  const float  * ptr_scales;
  float  * ptr_dst;
  int i;
  int w;
  long sample_count;
  short channel_count;
  // seek
  if(eep_seek(obj->eep, DATATYPE_EEG, from, 0)) {
    return NULL;
  }

  // convenience values
  channel_count = eep_get_chanc(obj->eep);
  sample_count = to - from;

  // get unscaled data
  buffer_unscaled = (sraw_t *)malloc(sizeof(sraw_t) * channel_count * sample_count);

  if(eep_read_sraw(obj->eep, DATATYPE_EEG, buffer_unscaled, sample_count)) {
    free(buffer_unscaled);
    return NULL;
  }
  // scale data
  buffer_scaled = (float *)malloc(sizeof(float) * channel_count * sample_count);
  ptr_src=buffer_unscaled;
  ptr_dst=buffer_scaled;
  ptr_scales = NULL;
  w = 0;
  i = 0;
  while(i<(channel_count * sample_count)) {
    if(!w) {
      w=channel_count;
      ptr_scales=obj->scales;
    }
    *ptr_dst = (float)(*ptr_src) * (*ptr_scales);
	++ptr_dst;
	++ptr_src;
	++ptr_scales;
    --w;
	++i;
  }
  free(buffer_unscaled);
  return buffer_scaled;
}
///////////////////////////////////////////////////////////////////////////////
float *
libeep_get_samples(cntfile_t handle, long from, long to) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  if(obj->data_type==dt_avr) {
    return _libeep_get_samples_avr(obj, from, to);
  }
  if(obj->data_type==dt_cnt) {
    return _libeep_get_samples_cnt(obj, from, to);
  }
  return NULL;
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_free_samples(float *buffer) {
  if(buffer) {
    free(buffer);
  }
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_add_samples(cntfile_t handle, const float *data, int n) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_write);
  sraw_t *buffer;
  const float  * ptr_src;
  sraw_t * ptr_dst;
  int c;

  c=CNTBUF_SIZE(obj->eep, n);
  buffer=(sraw_t*)malloc(c);
  ptr_src=data;
  ptr_dst=buffer;

  c/=sizeof(sraw_t);
  while(c--) {
    *ptr_dst++ = (sraw_t)(*ptr_src++ * SCALING_FACTOR);
  }

  eep_write_sraw(obj->eep, buffer, n);

  free(buffer);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_add_raw_samples(cntfile_t handle, const int32_t *data, int n) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_write);
  eep_write_sraw(obj->eep, data, n);
}
///////////////////////////////////////////////////////////////////////////////
int32_t *
libeep_get_raw_samples(cntfile_t handle, long from, long to) {
  sraw_t *buffer_unscaled;
  struct _libeep_entry * obj;

  obj = _libeep_get_object(handle, om_read);
  // seek
  if (eep_seek(obj->eep, DATATYPE_EEG, from, 0)) {
    return NULL;
  }
  // get unscaled data
  buffer_unscaled = (sraw_t *)malloc(CNTBUF_SIZE(obj->eep, to - from));
  if (eep_read_sraw(obj->eep, DATATYPE_EEG, buffer_unscaled, to - from)) {
    free(buffer_unscaled);
    return NULL;
  }
  return buffer_unscaled;
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_free_raw_samples(int32_t *buffer) {
  if(buffer) {
    free(buffer);
  }
}
///////////////////////////////////////////////////////////////////////////////
recinfo_t
libeep_create_recinfo() {
  return _libeep_recinfo_allocate();
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_add_recording_info(cntfile_t cnt_handle, recinfo_t recinfo_handle) {
  struct _libeep_entry * cnt = _libeep_get_object(cnt_handle, om_write);
  struct record_info_s * rec = _libeep_get_recinfo(recinfo_handle);

  // bail if this is not a cnt file
  if(cnt->data_type != dt_cnt) {
    return;
  }

  // bail if this is not a writable file
  if(cnt->open_mode != om_write) {
    return;
  }

  eep_set_recording_info(cnt->eep, rec);
}
///////////////////////////////////////////////////////////////////////////////
time_t
libeep_get_start_time(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_recording_startdate_epoch(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_get_start_date_and_fraction(recinfo_t handle, double* start_date, double* start_fraction) {
  record_info_t rec_inf;
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  if (start_date) *start_date = 0.0;
  if (start_fraction) *start_fraction = 0.0;
  if (eep_has_recording_info(obj->eep)) {
    eep_get_recording_info(obj->eep, &rec_inf);
    if (start_date) *start_date = rec_inf.m_startDate;
    if (start_fraction) *start_fraction = rec_inf.m_startFraction;
  }
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_start_time(recinfo_t handle, time_t start_time) {
  struct record_info_s * obj = _libeep_get_recinfo(handle);
  eep_unixdate_to_exceldate(start_time, &obj->m_startDate, &obj->m_startFraction);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_start_date_and_fraction(recinfo_t handle, double start_date, double start_fraction) {
  struct record_info_s * obj = _libeep_get_recinfo(handle);
  obj->m_startDate = start_date;
  obj->m_startFraction = start_fraction;
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_hospital(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_hospital(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_hospital(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szHospital) / sizeof(obj->m_szHospital[0]) - 1;
    strncpy(obj->m_szHospital, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_test_name(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_test_name(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_test_name(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szTestName) / sizeof(obj->m_szTestName[0]) - 1;
    strncpy(obj->m_szTestName, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_test_serial(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_test_serial(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_test_serial(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szTestSerial) / sizeof(obj->m_szTestSerial[0]) - 1;
    strncpy(obj->m_szTestSerial, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_physician(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_physician(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_physician(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szPhysician) / sizeof(obj->m_szPhysician[0]) - 1;
    strncpy(obj->m_szPhysician, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_technician(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_technician(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_technician(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szTechnician) / sizeof(obj->m_szTechnician[0]) - 1;
    strncpy(obj->m_szTechnician, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_machine_make(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_machine_make(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_machine_make(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szMachineMake) / sizeof(obj->m_szMachineMake[0]) - 1;
    strncpy(obj->m_szMachineMake, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_machine_model(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_machine_model(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_machine_model(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szMachineModel) / sizeof(obj->m_szMachineModel[0]) - 1;
    strncpy(obj->m_szMachineModel, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_machine_serial_number(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_machine_serial_number(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_machine_serial_number(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szMachineSN) / sizeof(obj->m_szMachineSN[0]) - 1;
    strncpy(obj->m_szMachineSN, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_patient_name(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_patient_name(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_patient_name(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szName) / sizeof(obj->m_szName[0]) - 1;
    strncpy(obj->m_szName, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_patient_id(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_patient_id(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_patient_id(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szID) / sizeof(obj->m_szID[0]) - 1;
    strncpy(obj->m_szID, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_patient_address(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_patient_address(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_patient_address(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szAddress) / sizeof(obj->m_szAddress[0]) - 1;
    strncpy(obj->m_szAddress, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_patient_phone(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_patient_phone(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_patient_phone(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szPhone) / sizeof(obj->m_szPhone[0]) - 1;
    strncpy(obj->m_szPhone, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_comment(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_comment(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_comment(recinfo_t handle, const char *value) {
  if (value) {
    struct record_info_s * obj = _libeep_get_recinfo(handle);
    const size_t len = sizeof(obj->m_szComment) / sizeof(obj->m_szComment[0]) - 1;
    strncpy(obj->m_szComment, value, len);
  }
}
///////////////////////////////////////////////////////////////////////////////
char
libeep_get_patient_sex(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_patient_sex(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_patient_sex(recinfo_t handle, char value) {
  struct record_info_s * obj = _libeep_get_recinfo(handle);
  obj->m_chSex = value;
}
///////////////////////////////////////////////////////////////////////////////
char
libeep_get_patient_handedness(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return eep_get_patient_handedness(obj->eep);
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_patient_handedness(recinfo_t handle, char value) {
  struct record_info_s * obj = _libeep_get_recinfo(handle);
  obj->m_chHandedness = value;
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_get_date_of_birth(cntfile_t handle, int * year, int * month, int  * day) {
  struct tm *dob = NULL;
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  dob = eep_get_patient_day_of_birth(obj->eep);
  *year = dob->tm_year + 1900;
  *month = dob->tm_mon + 1;
  *day = dob->tm_mday;
}
///////////////////////////////////////////////////////////////////////////////
void
libeep_set_date_of_birth(recinfo_t handle, int year, int month, int day) {
  struct record_info_s * obj = _libeep_get_recinfo(handle);
  struct tm temp;
  memset(&temp, 0, sizeof(temp));
  temp.tm_year = year - 1900;
  temp.tm_mon = month - 1;
  temp.tm_mday = day;

  // fill in blanks(tm_wday and tm_yday);
  mktime(&temp);

  memmove(&obj->m_DOB, &temp, sizeof(struct tm));
}
///////////////////////////////////////////////////////////////////////////////
int
libeep_add_trigger(cntfile_t handle, uint64_t sample, const char *code) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_write);
  return trg_set(eep_get_trg(obj->eep), sample, code);
}
///////////////////////////////////////////////////////////////////////////////
int
libeep_get_trigger_count(cntfile_t handle) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  return obj->processed_trigger_count;
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_trigger(cntfile_t handle, int idx, uint64_t *sample) {
  return libeep_get_trigger_with_extensions(handle, idx, sample, NULL);
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_trigger_with_extensions(cntfile_t handle, int idx, uint64_t *sample, struct libeep_trigger_extension * te) {
  struct _libeep_entry * obj = _libeep_get_object(handle, om_read);
  *sample = obj->processed_trigger_data[idx].sample;
  if(te != NULL) {
    te->type = obj->processed_trigger_data[idx].te.type;
    te->code = obj->processed_trigger_data[idx].te.code;
    te->duration_in_samples = obj->processed_trigger_data[idx].te.duration_in_samples;
    te->condition = obj->processed_trigger_data[idx].te.condition;
    te->description = obj->processed_trigger_data[idx].te.description;
    te->videofilename = obj->processed_trigger_data[idx].te.videofilename;
    te->impedances = obj->processed_trigger_data[idx].te.impedances;
  }
  return obj->processed_trigger_data[idx].label;
}
///////////////////////////////////////////////////////////////////////////////
long
libeep_get_zero_offset(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  if(obj->data_type==dt_avr) {
    return (int)(libeep_get_sample_frequency(handle) * eep_get_pre_stimulus_interval(obj->eep));
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_condition_label(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  if(obj->data_type==dt_avr) {
    return eep_get_conditionlabel(obj->eep);
  }
  return "none";
}
///////////////////////////////////////////////////////////////////////////////
const char *
libeep_get_condition_color(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  if(obj->data_type==dt_avr) {
    return eep_get_conditioncolor(obj->eep);
  }
  return "none";
}
///////////////////////////////////////////////////////////////////////////////
long
libeep_get_trials_total(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  if(obj->data_type==dt_avr) {
    return eep_get_total_trials(obj->eep);
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////
long
libeep_get_trials_averaged(cntfile_t handle) {
  struct _libeep_entry * obj=_libeep_get_object(handle, om_read);
  if(obj->data_type==dt_avr) {
    return eep_get_averaged_trials(obj->eep);
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////
chaninfo_t libeep_create_channel_info() {
  return _libeep_channels_allocate();
}
///////////////////////////////////////////////////////////////////////////////
void libeep_close_channel_info(chaninfo_t c) {
  _libeep_channels_free(c);
}
///////////////////////////////////////////////////////////////////////////////
int libeep_add_channel(chaninfo_t handle, const char *label, const char *ref_label, const char *unit) {
  eegchan_t *channels = NULL;
  const char *default_ref_label = "ref";
  const char *default_unit = "uV";
  struct _libeep_channels * obj = _libeep_get_channels(handle);
  // the channel label shall have a value; ref_label and unit might be NULL
  if (label == NULL) {
    return obj->count;
  }
  if (ref_label == NULL) {
    ref_label = default_ref_label;
  }
  if (unit == NULL) {
    unit = default_unit;
  }
  channels = (eegchan_t *)realloc(obj->channels, sizeof(eegchan_t) * (obj->count + 1));
  if (channels == NULL) {
    return obj->count;
  }
  obj->channels = channels;
  eep_chan_set(obj->channels, obj->count, label, 1, 1.0 / SCALING_FACTOR, unit);
  eep_chan_set_reflab(obj->channels, obj->count, ref_label);
  obj->count += 1;
  return obj->count;
}
