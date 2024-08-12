/*
**  OSSP val - Value Access
**  Copyright (c) 2002-2004 Ralf S. Engelschall <rse@engelschall.com>
**  Copyright (c) 2002-2004 The OSSP Project <http://www.ossp.org/>
**  Copyright (c) 2002-2004 Cable & Wireless <http://www.cw.com/>
**
**  This file is part of OSSP val, a value access library which
**  can be found at http://www.ossp.org/pkg/lib/val/.
**
**  Permission to use, copy, modify, and distribute this software for
**  any purpose with or without fee is hereby granted, provided that
**  the above copyright notice and this permission notice appear in all
**  copies.
**
**  THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
**  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
**  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
**  IN NO EVENT SHALL THE AUTHORS AND COPYRIGHT HOLDERS AND THEIR
**  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
**  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
**  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
**  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
**  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
**  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
**  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
**  SUCH DAMAGE.
**
**  val.h: library API
*/

#ifndef __VAL_H__
#define __VAL_H__

#include <stdarg.h>

/* maximum length of a structured value name */
#define VAL_MAXNAME 1024

/* the set of distinct value types */
enum {
    VAL_TYPE_VAL    = 1<<0,
    VAL_TYPE_PTR    = 1<<1,
    VAL_TYPE_CHAR   = 1<<2,
    VAL_TYPE_SHORT  = 1<<3,
    VAL_TYPE_INT    = 1<<4,
    VAL_TYPE_LONG   = 1<<5,
    VAL_TYPE_FLOAT  = 1<<6,
    VAL_TYPE_DOUBLE = 1<<7
};

/* the set of return codes */
typedef enum {
    VAL_OK,        /* everything ok */
    VAL_ERR_ARG,   /* error: invalid argument */
    VAL_ERR_USE,   /* error: invalid use */
    VAL_ERR_MEM,   /* error: no more memory */
    VAL_ERR_HSH,   /* error: hash table problem */
    VAL_ERR_INT,   /* error: internal error */
    VAL_ERR_SYS    /* error: system error (see errno) */
} val_rc_t;

/* the opaque data structure and type */
struct val_s;
typedef struct val_s val_t;

/* function type for use with val_apply() */
typedef val_rc_t (*val_cb_t)(void *, const char *, int, const char *, void *);

/* unique library identifier */
extern const char val_id[];

/* set of API functions */
val_rc_t val_create  (val_t **);
val_rc_t val_destroy (val_t *);
val_rc_t val_reg     (val_t *, const char *, int, const char *, void *);
val_rc_t val_unreg   (val_t *, const char *);
val_rc_t val_query   (val_t *, const char *, int *, char **, void **);
val_rc_t val_set     (val_t *, const char *, ...);
val_rc_t val_get     (val_t *, const char *, ...);
val_rc_t val_vset    (val_t *, const char *, va_list);
val_rc_t val_vget    (val_t *, const char *, va_list);
val_rc_t val_apply   (val_t *, const char *, int, val_cb_t, void *);

#endif /* __VAL_H__ */

