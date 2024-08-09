// system
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// libeep
#include <cnt/evt.h>
#include <eep/winsafe.h>
/******************************************************************************
 * misc
 *****************************************************************************/
#define TODO_MARKER { fprintf(stderr, "TODO: %s(%i): %s\n", __FILE__, __LINE__, __FUNCTION__); }
/*****************************************************************************/
enum _libeep_evt_log_level { evt_log_dbg, evt_log_inf, evt_log_err };
static
void
_libeep_evt_log(int level, int indent, const char * fmt, ...) {
  if(level >= evt_log_inf) {
    int     i;
    va_list args;

    for(i=0;i<indent;++i) {
      fprintf(stderr, "  ");
    }

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end (args);
  }
}
/*****************************************************************************/
static
void
_libeep_evt_string_delete(char * s) {
  if(s != NULL) {
    free(s);
  }
}
/*****************************************************************************/
static
void
_libeep_evt_string_append_string(char ** s1, const char * s2) {
  char   * tmp;
  size_t   len_s1=*s1 == NULL ? 0 : strlen(*s1);
  size_t   len_s2=s2 == NULL ? 0 : strlen(s2);

  tmp=(char *)malloc(len_s1 + len_s2 + 1);
  sprintf(tmp, "%s%s", *s1 == NULL ? "" : *s1, s2 == NULL ? "" : s2);

  _libeep_evt_string_delete(*s1);

  *s1 = tmp;
}
/*****************************************************************************/
static
void
_libeep_evt_string_append_float(char ** s1, const char * fmt, float f) {
  char tmp[16];
  snprintf(tmp, 16, fmt, f);
  _libeep_evt_string_append_string(s1, tmp);
}
/******************************************************************************
 * libeep evt class
 *****************************************************************************/
libeep_evt_class_t *
libeep_evt_class_new() {
  libeep_evt_class_t * rv = NULL;

  rv = (libeep_evt_class_t *)malloc(sizeof(libeep_evt_class_t));
  memset(rv, 0, sizeof(libeep_evt_class_t));

  return rv;
}
/*****************************************************************************/
void
libeep_evt_class_delete(libeep_evt_class_t * c) {
  if(c != NULL) {
    _libeep_evt_string_delete(c->name);
    free(c);
  }
}
/******************************************************************************
 * libeep evt
 *****************************************************************************/
libeep_evt_t * libeep_evt_new() {
  libeep_evt_t * rv = NULL;

  rv=(libeep_evt_t *)malloc(sizeof(libeep_evt_t));
  memset(rv, 0, sizeof(libeep_evt_t));

  return rv;
}
/*****************************************************************************/
void libeep_evt_delete(libeep_evt_t * e) {
  if(e != NULL) {
    while(e->evt_list_first) {
      libeep_evt_event_t * tmp = e->evt_list_first;
      e->evt_list_first = e->evt_list_first->next_event;
      libeep_evt_event_delete(tmp);
    }
    free(e);
  }
}
/******************************************************************************
 * internal function; add to list
 *****************************************************************************/
static
void
_libeep_evt_event_list_add_back(libeep_evt_t * l, libeep_evt_event_t *n) {
  if(l->evt_list_first == NULL) {
    l->evt_list_first = n;
    l->evt_list_last = n;
  } else {
    l->evt_list_last->next_event = n;
    l->evt_list_last             = n;
  }
}
/******************************************************************************
 * internal function; read string
 *****************************************************************************/
static
char *
_libeep_evt_read_string(FILE * f) {
  char    * rv = NULL;
  size_t    length = 0;
  uint8_t   byte = 0;
  uint16_t  word = 0;
  uint32_t  dword = 0;
  uint64_t  qword = 0;

  /* is the length encoded in a byte? */
  if(fread(&byte, sizeof(uint8_t), 1, f) == 1) {
    length = (size_t)byte;
    if(byte == 0xFF) {
      /* is the size encoded in a word? */
      if(fread(&word, sizeof(uint16_t), 1, f) == 1) {
        length = (size_t)word;
        if(word == 0xFFFF) {
          /* is the size encoded in a dword? */
          if(fread(&dword, sizeof(uint32_t), 1, f) == 1) {
            length = (size_t)dword;
            if(dword == 0xFFFFFFFF) {
              /* is the size encoded in a qword? */
              if(fread(&qword, sizeof(uint64_t), 1, f) == 1) {
                length = (size_t)qword;
              }
            }
          }
        }
      }
    }
  }

  if(length == 0) {
    return NULL;
  }

  rv=(char *)malloc(length + 1);
  memset(rv, 0, length + 1);
  if(fread(rv, length, 1, f) == 1) {
  } else {
    /* something went wrong reading the string */
    TODO_MARKER;
  }

  return rv;
}
/*****************************************************************************/
struct char_pair {
  char lo;
  char hi;
};
typedef struct char_pair char_pair_t;
/******************************************************************************
 * internal function; read wstring
 * does a rough conversion from wchar_t * to char * by ingnoring high byte.
 *****************************************************************************/
static
char *
_libeep_evt_read_wstring(FILE * f) {
  char    * rv;
  int32_t   length;
  int32_t   bytes;
  int32_t   i;

  rv = NULL;
  if(fread(&bytes, sizeof(int32_t), 1, f) == 1) {
    // character pair buffer
    char_pair_t * cp_array;
    cp_array = (char_pair_t *)malloc(bytes);
    fread(cp_array, bytes, 1, f);

    // allocate return value
    length = bytes / 2;
    rv = (char *)malloc(length + 1);
    memset(rv, 0, length + 1);

    // convert
    for(i=0;i<length;++i) {
      rv[i] = cp_array[i].lo;
    }

    // free
    free(cp_array);
  } else {
    /* something went wrong reading the string */
    TODO_MARKER;
  }

  return rv;
}
/******************************************************************************
 * internal function; read evt class
 *****************************************************************************/
static
libeep_evt_class_t *
_libeep_evt_read_class(FILE * f, int indent) {
  libeep_evt_class_t * rv = libeep_evt_class_new();

  if(fread(&rv->tag, sizeof(int32_t), 1, f) == 1) {
    switch(rv->tag) {
      case 0:
        break;
      case -1: /* string */
        rv->name = _libeep_evt_read_string(f);
        break;
      default: /* unicode */
        break;
    }
  }

  _libeep_evt_log(evt_log_dbg, indent, "%s: class(%i, %s)\n", __FUNCTION__, rv->tag, rv->name);

  return rv;
}
/******************************************************************************
 * libeep_evt event base
 *****************************************************************************/
libeep_evt_GUID_t *
libeep_evt_GUID_new() {
  libeep_evt_GUID_t * rv = NULL;

  rv = (libeep_evt_GUID_t *)malloc(sizeof(libeep_evt_GUID_t));
  memset(rv, 0, sizeof(libeep_evt_GUID_t));

  return rv;
}
/*****************************************************************************/
void
libeep_evt_GUID_delete(libeep_evt_GUID_t * g) {
  if(g != NULL) {
    free(g);
  }
}
/******************************************************************************
 * libeep_evt event base
 *****************************************************************************/
libeep_evt_event_t *
libeep_evt_event_new() {
  libeep_evt_event_t * rv = NULL;

  rv = (libeep_evt_event_t *)malloc(sizeof(libeep_evt_event_t));
  memset(rv, 0, sizeof(libeep_evt_event_t));

  return rv;
}
/*****************************************************************************/
void
libeep_evt_event_delete(libeep_evt_event_t * e) {
  if(e != NULL) {
    libeep_evt_GUID_delete(e->guid);
    _libeep_evt_string_delete(e->unused_name);
    _libeep_evt_string_delete(e->unused_user_visible_name);
    _libeep_evt_string_delete(e->condition);
    _libeep_evt_string_delete(e->description);
    _libeep_evt_string_delete(e->videofilename);
    _libeep_evt_string_delete(e->impedances);
    free(e);
  }
}
/******************************************************************************
 * internal enum; read evt events
 *****************************************************************************/
enum vt_e {
  vt_empty = 0,
  vt_null = 1,
  vt_i2 = 2,
  vt_i4 = 3,
  vt_r4 = 4,
  vt_r8 = 5,
  vt_bstr = 8,
  vt_bool = 11,
  vt_u4 = 19,
  vt_array = 0x2000,
  vt_byref = 0x4000,
};
struct libeep_evt_variant {
  enum vt_e   type;
  int16_t     i16;
  int32_t     i32;
  float       f;
  uint32_t    u32;
  float     * f_array;
  uint32_t    f_array_size;
  double      d;
  char      * string;
};
typedef struct libeep_evt_variant libeep_evt_variant_t;
/*****************************************************************************/
static
void
_libeep_evt_read_variant_base(FILE *f, int indent, libeep_evt_variant_t * variant) {
  if(fread(&variant->type, sizeof(int16_t), 1, f) == 1) {
    _libeep_evt_log(evt_log_dbg, indent, "%s: type: %i\n", __FUNCTION__, variant->type);

    switch(variant->type) {
      case vt_empty:
      case vt_null:
        break;
      case vt_i2:
        fread(&variant->i16, sizeof(int16_t), 1, f);
        _libeep_evt_log(evt_log_dbg, indent, "%s: i2: %i\n", __FUNCTION__, variant->i16);
        break;
      case vt_i4:
        fread(&variant->i32, sizeof(int32_t), 1, f);
        _libeep_evt_log(evt_log_dbg, indent, "%s: i4: %i\n", __FUNCTION__, variant->i32);
        break;
      case vt_r4:
        fread(&variant->f, sizeof(float), 1, f);
        _libeep_evt_log(evt_log_dbg, indent, "%s: r4: %i\n", __FUNCTION__, variant->f);
        break;
      case vt_r8:
        fread(&variant->d, sizeof(double), 1, f);
        _libeep_evt_log(evt_log_dbg, indent, "%s: r8: %g\n", __FUNCTION__, variant->d);
        break;
      case vt_bstr:
        variant->string = _libeep_evt_read_wstring(f);
        _libeep_evt_log(evt_log_dbg, indent, "%s: wstring: %ls\n", __FUNCTION__, variant->string);
        break;
      case vt_bool:
        TODO_MARKER;
        break;
      case vt_u4:
        fread(&variant->u32, sizeof(uint32_t), 1, f);
        _libeep_evt_log(evt_log_dbg, indent, "%s: i4: %i\n", __FUNCTION__, variant->i32);
        break;
      case vt_array:
        break;
      case vt_byref:
        break;
    }
  }
}
/*****************************************************************************/
static
void
_libeep_evt_read_variant_array(FILE *f, int indent, libeep_evt_variant_t * outer_variant) {
  libeep_evt_variant_t variant;
  memset(&variant, 0, sizeof(libeep_evt_variant_t));

  _libeep_evt_read_variant_base(f, indent + 1, &variant);

  _libeep_evt_log(evt_log_dbg, indent, "%s: variant.type: %i\n", __FUNCTION__, variant.type);

  switch(variant.type) {
    case vt_empty:
    case vt_null:
      break;
    case vt_i2:
      TODO_MARKER;
      break;
    case vt_i4:
      TODO_MARKER;
      break;
    case vt_r4:
      assert(outer_variant->f_array == NULL);
      fread(&outer_variant->f_array_size, sizeof(uint32_t), 1, f);
      _libeep_evt_log(evt_log_dbg, indent, "%s: outer_variant->f_array_size: %i\n", __FUNCTION__, outer_variant->f_array_size);
      outer_variant->f_array=(float *)malloc(sizeof(float) * outer_variant->f_array_size);
      fread(outer_variant->f_array, sizeof(float) * outer_variant->f_array_size, 1, f);
      break;
    case vt_r8:
      TODO_MARKER;
      break;
    case vt_bstr:
      TODO_MARKER;
      break;
    case vt_bool:
      TODO_MARKER;
      break;
    case vt_u4:
      TODO_MARKER;
      break;
    case vt_array:
      break;
    case vt_byref:
      break;
  }
}
/*****************************************************************************/
static
void
_libeep_evt_read_variant(FILE *f, int indent, libeep_evt_variant_t * variant) {
  _libeep_evt_read_variant_base(f, indent + 1, variant);

  switch(variant->type) {
    case vt_i2:
    case vt_i4:
    case vt_r4:
    case vt_r8:
    case vt_bool:
    case vt_bstr:
      break;
    default:
      if( (variant->type & vt_byref) || (variant->type & vt_array) ) {
        _libeep_evt_read_variant_array(f, indent + 1, variant);
      }
      break;
  }
}
/*****************************************************************************/
static
void
_libeep_evt_event_process_variant(libeep_evt_event_t * ev, libeep_evt_variant_t * variant, const char * descriptor_name) {
  if(!strcmp(descriptor_name, "EventCode")) {
    ev->code = variant->i32;
  } else if(!strcmp(descriptor_name, "Condition")) {
    if(variant->string) {
      ev->condition=strdup(variant->string);
    }
  } else if(!strcmp(descriptor_name, "VideoFileName")) {
    if(variant->string) {
      ev->videofilename=strdup(variant->string);
    }
  } else if(!strcmp(descriptor_name, "Impedance")) {
    uint32_t n;
    for(n=0;n<variant->f_array_size;++n) {
      if(n) {
        _libeep_evt_string_append_float(&ev->impedances, " %f", variant->f_array[n]);
      } else {
        _libeep_evt_string_append_float(&ev->impedances, "%f", variant->f_array[n]);
      }
    }
  } else if(!strcmp(descriptor_name, "Name:")) {
    assert(ev->description == NULL);
    ev->description=strdup(variant->string);
  } else {
    char tmp[1024];
    switch(variant->type) {
      case vt_empty:
      case vt_null:
        break;
      case vt_i2:
        snprintf(tmp, 1024, " %s=%d", descriptor_name, variant->i16);
        break;
      case vt_i4:
        snprintf(tmp, 1024, " %s=%d", descriptor_name, variant->i32);
        break;
      case vt_r4:
        snprintf(tmp, 1024, " %s=%f", descriptor_name, variant->f);
        break;
      case vt_r8:
        snprintf(tmp, 1024, " %s=%f", descriptor_name, variant->d);
        break;
      case vt_u4:
        snprintf(tmp, 1024, " %s=%u", descriptor_name, variant->u32);
        break;
      case vt_bstr:
        snprintf(tmp, 1024, " %s=%s", descriptor_name, variant->string);
        break;
      case vt_array:
      case vt_byref:
      case vt_bool:
      break;
    }
    _libeep_evt_string_append_string(&ev->condition, tmp);
  }
}
/*****************************************************************************/
static
void
_libeep_evt_read_epoch_descriptors(FILE *f, int indent, libeep_evt_event_t * ev) {
  int32_t i = 0;
  int32_t size = 0;

  if(fread(&size, sizeof(int32_t), 1, f) == 1) {
    for(i=0;i<size;++i) {
      libeep_evt_variant_t   variant;
      char                 * descriptor_name;
      char                 * descriptor_unit;

      memset(&variant, 0, sizeof(libeep_evt_variant_t));

      descriptor_name = _libeep_evt_read_string(f);
      _libeep_evt_log(evt_log_dbg, indent, "%s: descriptor_name: %s\n", __FUNCTION__, descriptor_name);

      _libeep_evt_read_variant(f, indent + 1, &variant);

      descriptor_unit = _libeep_evt_read_string(f);
      _libeep_evt_log(evt_log_dbg, indent, "%s: descriptor_unit: %s\n", __FUNCTION__, descriptor_unit);

      _libeep_evt_event_process_variant(ev, &variant, descriptor_name);

      _libeep_evt_string_delete(descriptor_name);
      _libeep_evt_string_delete(descriptor_unit);
      _libeep_evt_string_delete(variant.string);
      if(variant.f_array != NULL) {
        free(variant.f_array);
      }
    }
  }
}
/*****************************************************************************/
static
libeep_evt_GUID_t *
_libeep_evt_read_GUID(FILE * f, int indent) {
  libeep_evt_GUID_t * rv = NULL;

  rv = libeep_evt_GUID_new();

  if(fread(rv, sizeof(libeep_evt_GUID_t), 1, f) == 1) {
    _libeep_evt_log(evt_log_dbg, indent, "%s: GUID %i %i %i [%i %i %i %i %i %i %i %i] ....:\n", __FUNCTION__, rv->data1, rv->data2, rv->data3, rv->data4[0], rv->data4[1], rv->data4[2], rv->data4[3], rv->data4[4], rv->data4[5], rv->data4[6], rv->data4[7]);
  }
  return rv;
}
/*****************************************************************************/
static
void
_libeep_evt_read_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  libeep_evt_class_t * clss = NULL;

  if(fread(&ev->visible_id, sizeof(int32_t), 1, f) == 1) {
    ev->guid = _libeep_evt_read_GUID(f, indent + 1);
    clss = _libeep_evt_read_class(f, indent + 1);
    libeep_evt_class_delete(clss);
    ev->unused_name = _libeep_evt_read_string(f);
    if(e->header.version >= 78) {
      ev->unused_user_visible_name = _libeep_evt_read_string(f);
    }
    if(fread(&ev->type, sizeof(int32_t), 1, f) == 1) {
      if(fread(&ev->state, sizeof(int32_t), 1, f) == 1) {
        if(fread(&ev->original, sizeof(int8_t), 1, f) == 1) {
          if(fread(&ev->duration, sizeof(double), 1, f) == 1) {
            if(fread(&ev->duration_offset, sizeof(double), 1, f) == 1) {
              if(fread(&ev->time_stamp.date, sizeof(double), 1, f) == 1) {
                if(fread(&ev->time_stamp.fraction, sizeof(double), 1, f) == 1) {
                  _libeep_evt_log(evt_log_dbg, indent, "%s: type: %i state: %i original: %i duration: %g offset: %g date: %g fraction: %g\n", __FUNCTION__, ev->type, ev->state, ev->original, ev->duration, ev->duration_offset, ev->time_stamp.date, ev->time_stamp.fraction);

                  if(e->header.version >= 11 && e->header.version < 19) {
                    TODO_MARKER;
                  }
                  if(e->header.version >= 19) {
                    _libeep_evt_read_epoch_descriptors(f, indent + 1, ev);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/*****************************************************************************/
static
void
_libeep_evt_read_channel_info(FILE * f, int indent) {
  char * channel_active;
  char * channel_reference;

  channel_active = _libeep_evt_read_string(f);
  channel_reference = _libeep_evt_read_string(f);

  _libeep_evt_log(evt_log_dbg, indent, "%s: channel_active: %s\n", __FUNCTION__, channel_active);
  _libeep_evt_log(evt_log_dbg, indent, "%s: channel_reference: %s\n", __FUNCTION__, channel_reference);

  _libeep_evt_string_delete(channel_active);
  _libeep_evt_string_delete(channel_reference);
}
/*****************************************************************************/
static
void
_libeep_evt_read_epoch_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  _libeep_evt_read_event(f, indent + 1, e, ev);
  if(e->header.version < 33) {
    int32_t tmp;
    fread(&tmp, sizeof(int32_t), 1, f);
    _libeep_evt_log(evt_log_dbg, indent, "%s: tmp: %s\n", __FUNCTION__, tmp);
  }
}
/*****************************************************************************/
static
void
_libeep_evt_read_event_marker(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  int32_t   show_amplitude = 0;
  int8_t    show_duration = 0;

  _libeep_evt_read_event(f, indent + 1, e, ev);
  _libeep_evt_read_channel_info(f, indent + 1);
  assert(ev->description==NULL);
  ev->description = _libeep_evt_read_string(f);
  if(e->header.version >= 35) {
    if(e->header.version >= 103) {
      fread(&show_amplitude, sizeof(int32_t), 1, f);
    } else {
      int8_t tmp;
      fread(&tmp, sizeof(int8_t), 1, f);
      show_amplitude = tmp;
    }
    fread(&show_duration, sizeof(int8_t), 1, f);
  }
  _libeep_evt_log(evt_log_dbg, indent, "%s: show_amplitude: %i\n", __FUNCTION__, show_amplitude);
  _libeep_evt_log(evt_log_dbg, indent, "%s: show_duration: %i\n", __FUNCTION__, show_duration);
  _libeep_evt_log(evt_log_dbg, indent, "%s: ev->description: %s\n", __FUNCTION__, ev->description);
}
/*****************************************************************************/
static
void
_libeep_evt_read_artefact_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  _libeep_evt_read_event(f, indent, e, ev);
  _libeep_evt_read_channel_info(f, indent + 1);
  if(e->header.version >= 174) {
    ev->description = _libeep_evt_read_string(f);
    _libeep_evt_log(evt_log_dbg, indent, "%s: ev->description: %s\n", __FUNCTION__, ev->description);
  }
}
/*****************************************************************************/
static
void
_libeep_evt_read_spike_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  _libeep_evt_read_event(f, indent + 1, e, ev);
  TODO_MARKER;
}
/*****************************************************************************/
static
void
_libeep_evt_read_seizure_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  _libeep_evt_read_event(f, indent + 1, e, ev);
  TODO_MARKER;
}
/*****************************************************************************/
static
void
_libeep_evt_read_sleep_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  _libeep_evt_read_event(f, indent + 1, e, ev);
  TODO_MARKER;
}
/*****************************************************************************/
static
void
_libeep_evt_read_rpeak_event(FILE * f, int indent, libeep_evt_t * e, libeep_evt_event_t * ev) {
  _libeep_evt_read_event(f, indent + 1, e, ev);
  TODO_MARKER;
}
/*****************************************************************************/
void
libeep_evt_header_print(const libeep_evt_header_t * h) {
  _libeep_evt_log(evt_log_inf, 0, "header:\n");
  _libeep_evt_log(evt_log_inf, 0, "  ctime: %i\n", h->ctime);
  _libeep_evt_log(evt_log_inf, 0, "  mtime: %i\n", h->mtime);
  _libeep_evt_log(evt_log_inf, 0, "  atime: %i\n", h->atime);
  _libeep_evt_log(evt_log_inf, 0, "  version: %i\n", h->version);
  _libeep_evt_log(evt_log_inf, 0, "  comp.mode: %i\n", h->compression_mode);
  _libeep_evt_log(evt_log_inf, 0, "  encr.mode: %i\n", h->encryption_mode);
}
/*****************************************************************************/
void
libeep_evt_event_print(const libeep_evt_event_t * e) {
  _libeep_evt_log(evt_log_inf, 0, "libeep_evt_event_t {\n");
  _libeep_evt_log(evt_log_inf, 0, "  visible_id......... %i\n", e->visible_id);
  if(e->guid) {
    _libeep_evt_log(evt_log_inf, 0, "  GUID............... %i %i %i [%i %i %i %i %i %i %i %i]\n", e->guid->data1, e->guid->data2, e->guid->data3, e->guid->data4[0], e->guid->data4[1], e->guid->data4[2], e->guid->data4[3], e->guid->data4[4], e->guid->data4[5], e->guid->data4[6], e->guid->data4[7]);
  }
  _libeep_evt_log(evt_log_inf, 0, "  name............... %s\n", e->unused_name);
  _libeep_evt_log(evt_log_inf, 0, "  user_visible_name.. %s\n", e->unused_user_visible_name);
  _libeep_evt_log(evt_log_inf, 0, "  type............... %i\n", e->type);
  _libeep_evt_log(evt_log_inf, 0, "  state.............. %i\n", e->state);
  _libeep_evt_log(evt_log_inf, 0, "  original........... %i\n", e->original);
  _libeep_evt_log(evt_log_inf, 0, "  duration........... %g\n", e->duration);
  _libeep_evt_log(evt_log_inf, 0, "  duration_offset.... %g\n", e->duration_offset);
  _libeep_evt_log(evt_log_inf, 0, "  timestamp.......... %8.8f / %8.8f\n", e->time_stamp.date, e->time_stamp.fraction);
  _libeep_evt_log(evt_log_inf, 0, "  code............... %i\n", e->code);
  _libeep_evt_log(evt_log_inf, 0, "  condition.......... %s\n", e->condition);
  _libeep_evt_log(evt_log_inf, 0, "  description........ %s\n", e->description);
  _libeep_evt_log(evt_log_inf, 0, "  videofilename...... %s\n", e->videofilename);
  _libeep_evt_log(evt_log_inf, 0, "  impedances......... %s\n", e->impedances);
  _libeep_evt_log(evt_log_inf, 0, "}\n");
}
/******************************************************************************
 * internal function; read evt library
 *****************************************************************************/
static
void
_libeep_evt_read_library(FILE * f, int indent, libeep_evt_t * e) {
  char               * library_name;
  uint32_t             length;
  uint32_t             i;
  libeep_evt_class_t * clss;

  library_name = _libeep_evt_read_string(f);

  _libeep_evt_log(evt_log_dbg, indent, "%s: library_name: %s\n", __FUNCTION__, library_name);

  _libeep_evt_string_delete(library_name);

  if(fread(&length, sizeof(uint32_t), 1, f) == 1) {
    _libeep_evt_log(evt_log_dbg, indent, "%s: %u library items\n", __FUNCTION__, length);

    for(i=0;i<length;++i) {
      libeep_evt_event_t * ev = libeep_evt_event_new();

      long offset = ftell(f);

      clss = _libeep_evt_read_class(f, indent + 1);

      _libeep_evt_log(evt_log_dbg, indent, "%s: i: %u offset: 0x%lx(%ld) class(%i, %s)\n", __FUNCTION__, i, offset, offset, clss->tag, clss->name);

      if(clss->name) {
        if (!strcmp(clss->name, "class dcEpochEvent_c"))         { _libeep_evt_read_epoch_event(f, indent + 1, e, ev); }
        else if (!strcmp(clss->name, "class dcEventMarker_c"))   { _libeep_evt_read_event_marker(f, indent + 1, e, ev); }
        else if (!strcmp(clss->name, "class dcArtefactEvent_c")) { _libeep_evt_read_artefact_event(f, indent + 1, e, ev); }
        else if (!strcmp(clss->name, "class dcSpikeEvent_c"))    { _libeep_evt_read_spike_event(f, indent + 1, e, ev); }
        else if (!strcmp(clss->name, "class dcSeizureEvent_c"))  { _libeep_evt_read_seizure_event(f, indent + 1, e, ev); }
        else if (!strcmp(clss->name, "class dcSleepEvent_c"))    { _libeep_evt_read_sleep_event(f, indent + 1, e, ev); }
        else if (!strcmp(clss->name, "class dcRPeakEvent_c"))    { _libeep_evt_read_rpeak_event(f, indent + 1, e, ev); }
        else {
          _libeep_evt_log(evt_log_err, indent, "unknown class: %s\n", clss->name);
        }
      }

      libeep_evt_class_delete(clss);
      _libeep_evt_event_list_add_back(e, ev);
    }
  }
}
/******************************************************************************
 * internal function; read evt file
 *****************************************************************************/
static
void
_libeep_evt_read_file(FILE * f, int indent, libeep_evt_t * e) {
  libeep_evt_class_t * clss;

  if(fread(&e->header, sizeof(libeep_evt_header_t), 1, f) != 1) {
    return;
  }

  clss = _libeep_evt_read_class(f, indent + 1);
  if(clss && clss->tag==-1 && !strcmp("class dcEventsLibrary_c", clss->name)) {
    _libeep_evt_read_library(f, indent + 1, e);
  }
  libeep_evt_class_delete(clss);
}
/******************************************************************************
 * evt io
 *****************************************************************************/
libeep_evt_t *
libeep_evt_read(const char * filename) {
  libeep_evt_t * rv;
  FILE         * f;

  f = fopen(filename, "rb");
  if(f == NULL) {
    return NULL;
  }

  rv = libeep_evt_new();
  _libeep_evt_read_file(f, 0, rv);

  fclose(f);

  return rv;
}
