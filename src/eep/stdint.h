#ifndef __libeep_std_int_h__
#define __libeep_std_int_h__

#if defined(_MSC_VER) && (_MSC_VER <= 1500)
	typedef __int32 int32_t;
	typedef unsigned __int32 uint32_t;
	typedef __int64 int64_t;
	typedef unsigned __int64 uint64_t;
#else
	#include <stdint.h>
#endif

#endif
