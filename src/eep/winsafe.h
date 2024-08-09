#ifndef __libeep_eep_winsafe_h__
#define __libeep_eep_winsafe_h__

#if defined(_MSC_VER)
  #define snprintf _snprintf
  #define strncasecmp _strnicmp
  #define strcasecmp _stricmp
#endif

#endif
