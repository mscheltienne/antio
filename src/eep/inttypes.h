#ifndef __libeep_eep_int_types_h__
#define __libeep_eep_int_types_h__

#ifdef WIN32
  #define PRIu64       "I64u"
  #define SCNd64       "I64d"
  #define SCNu64       "I64u"
#else
  #include <inttypes.h>
#endif

#endif
