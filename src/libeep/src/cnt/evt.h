#ifndef __libeep_cnt_evt_h__
#define __libeep_cnt_evt_h__

// system
#include <stdint.h>

/********************************************************************************
 * libeep_evt header
 ********************************************************************************/
struct libeep_evt_header {
  uint32_t ctime;
  uint32_t mtime;
  uint32_t atime;
   int32_t version;
   int32_t compression_mode;
   int32_t encryption_mode;
};
typedef struct libeep_evt_header libeep_evt_header_t;

/********************************************************************************
 * libeep_evt class
 ********************************************************************************/
struct libeep_evt_class {
  int32_t   tag;
  char    * name;
};
typedef struct libeep_evt_class libeep_evt_class_t;

libeep_evt_class_t * libeep_evt_class_new();
void                 libeep_evt_class_delete(libeep_evt_class_t *);

/********************************************************************************
 * libeep_evt event base
 ********************************************************************************/
struct libeep_evt_GUID {
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t  data4[8];
};
typedef struct libeep_evt_GUID libeep_evt_GUID_t;

libeep_evt_GUID_t * libeep_evt_GUID_new();
void                libeep_evt_GUID_delete(libeep_evt_GUID_t *);

struct double_date {
  double date;
  double fraction;
};
typedef struct double_date double_date_t;

/********************************************************************************
 * libeep_evt event
 ********************************************************************************/
struct libeep_evt_event {
  int32_t             visible_id;
  libeep_evt_GUID_t * guid;

  char              * unused_name;
  char              * unused_user_visible_name;

  int32_t             type;
  int32_t             state;
  int8_t              original;
  double              duration;
  double              duration_offset;
  double_date_t       time_stamp;

  int32_t             code;
  char              * description;
  char              * condition;
  char              * videofilename;
  char              * impedances;

  /* linked list */
  struct libeep_evt_event * next_event;
};
typedef struct libeep_evt_event libeep_evt_event_t;

libeep_evt_event_t * libeep_evt_event_new();
void                 libeep_evt_event_delete(libeep_evt_event_t *);

/********************************************************************************
 * libeep_evt
 ********************************************************************************/
struct libeep_evt {
  libeep_evt_header_t   header;
  libeep_evt_event_t  * evt_list_first;
  libeep_evt_event_t  * evt_list_last;
};
typedef struct libeep_evt libeep_evt_t;

libeep_evt_t * libeep_evt_new();
void           libeep_evt_delete(libeep_evt_t *);

/********************************************************************************
 * evt io
 ********************************************************************************/
libeep_evt_t * libeep_evt_read(const char *);

/********************************************************************************
 * print stuff
 ********************************************************************************/
void libeep_evt_header_print(const libeep_evt_header_t *);
void libeep_evt_event_print(const libeep_evt_event_t *);

#endif
