#ifndef __libeep_cnt_seg_h__
#define __libeep_cnt_seg_h__

#include <stdint.h>

/**
  *
  */
struct libeep_seg_info {
  double   date;
  double   fraction;
  uint32_t sample_count;
};
typedef struct libeep_seg_info libeep_seg_info_t;
/**
  *
  */
struct libeep_seg {
  int                 count;
  libeep_seg_info_t * array;
};
typedef struct libeep_seg libeep_seg_t;
/**
 *
 */
libeep_seg_t * libeep_seg_read(const char *);
void           libeep_seg_delete(libeep_seg_t *);

#endif
