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
**  val.c: library implementation
*/

/* include system API headers */
#include <stdio.h>       /* for "size_t" */
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

/* optional memory debugging support */
#if defined(HAVE_DMALLOC_H) && defined(WITH_DMALLOC)
#include "dmalloc.h"
#endif

/* include own API header */
#include <eep/val.h>

/* boolean values */
#ifndef FALSE
#define FALSE (0)
#endif
#ifndef TRUE
#define TRUE  (!FALSE)
#endif

/* unique library identifier */
const char val_id[] = "OSSP val";

/* support for OSSP ex based exception throwing */
#ifdef WITH_EX
#include "ex.h"
#define VAL_RC(rv) \
    (  (rv) != VAL_OK && (ex_catching && !ex_shielding) \
     ? (ex_throw(val_id, NULL, (rv)), (rv)) : (rv) )
#else
#define VAL_RC(rv) (rv)
#endif /* WITH_EX */

/*
**  ____                            ____
**  ____ LINEAR HASHING SUB-LIBRARY ____
**
**  This part implements a Dynamic Hash Table, based on WITOLD LITWIN
**  and PER-AKE LARSON's ``Linear Hashing'' algorithm (1980/1988),
**  implemented on top of a two-level virtual array with separate
**  collision chains as the backend data structure. Some ideas were
**  taken over from MIKAEL PETTERSON's Linear Hashing enhancements
**  (1993). The code is derived from OSSP act. See there for details.
*/

struct lh_st;
typedef struct lh_st lh_t;

typedef int (*lh_cb_t)(void *ctx, const void *keyptr, int keylen, const void *datptr, int datlen);

/* fixed size (number of pointers) of the directory and of each segment */
#define INITDIRSIZE  256 /* can be an arbitrary value */
#define SEGMENTSIZE  512 /* has to be a power of 2 for below arithmetic */

/* the borders for the hash table load */
#define MINLOADFCTR  1   /* should be between 0 and 1 */
#define MAXLOADFCTR  2   /* should be between 2 and 4 */

/* the per-element structure (keep as small as possible!) */
typedef struct element_st element_t;
struct element_st {
    element_t    *e_next;    /* pointer to next element in collision chain */
    unsigned long e_hash;    /* cached hash value of element (rehashing optimization) */
    void         *e_keyptr;  /* pointer to key (= also pointer to key+data memory chunk) */
    void         *e_datptr;  /* pointer to data in key+data memory chunk */
    void         *e_endptr;  /* pointer to end of key+data memory chunk */
};

/* the hash table segments */
typedef struct segment_st segment_t;
struct segment_st {
    element_t *s_element[SEGMENTSIZE]; /* array of pointers to elements */
};

/* the top-level hash table structure */
struct lh_st {
    unsigned int   h_p;         /* pointer to start of unsplit region */
    unsigned int   h_pmax;      /* pointer to end of unsplit region */
    int            h_slack;     /* grow/shrink indicator */
    unsigned       h_dirsize;   /* current size of directory */
    segment_t    **h_dir;       /* pointer to directory */
};

/* on-the-fly calculate index into directory and segment from virtual array index */
#define DIRINDEX(addr) (int)((addr) / SEGMENTSIZE)
#define SEGINDEX(addr) (int)((addr) % SEGMENTSIZE)

/* on-the-fly calculate lengths of key and data to reduce memory in element_t */
#define el_keylen(el) ((char *)((el)->e_endptr)-(char *)((el)->e_keyptr))
#define el_datlen(el) ((char *)((el)->e_keyptr)-(char *)((el)->e_datptr))

/*
 * BJDDJ Hash Function (Bob Jenkins, Dr. Dobbs Journal):
 * This is a very complex but also very good hash function, as proposed
 * in the March'97 issue of Dr. Dobbs Journal (DDJ) by Bob Jenkins (see
 * http://burtleburtle.net/bob/hash/doobs.html for online version). He
 * showed that this hash function has both very good distribution and
 * performance and our own hash function comparison confirmed this. The
 * only difference to the original function of B.J. here is that our
 * version doesn't provide the `level' (= previous hash) argument for
 * consistency reasons with the other hash functions (i.e. same function
 * signature). It can be definetely recommended as a good general
 * purpose hash function.
 */
static long
lh_hash(
    register const unsigned char *k,
    register size_t length)
{
    register long a,b,c,len;

    /* some abbreviations */
#define ub4 long
#define mix(a,b,c) { \
        a -= b; a -= c; a ^= (c>>13); \
        b -= c; b -= a; b ^= (a<< 8); \
        c -= a; c -= b; c ^= (b>>13); \
        a -= b; a -= c; a ^= (c>>12); \
        b -= c; b -= a; b ^= (a<<16); \
        c -= a; c -= b; c ^= (b>> 5); \
        a -= b; a -= c; a ^= (c>> 3); \
        b -= c; b -= a; b ^= (a<<10); \
        c -= a; c -= b; c ^= (b>>15); \
    }

    /* setup the internal state */
    len = length;
    a = b = 0x9e3779b9;  /* the golden ratio; an arbitrary value */
    c = 0;

    /* handle most of the key */
    while (len >= 12) {
        a += (k[0] +((ub4)k[1]<<8) +((ub4)k[ 2]<<16) +((ub4)k[ 3]<<24));
        b += (k[4] +((ub4)k[5]<<8) +((ub4)k[ 6]<<16) +((ub4)k[ 7]<<24));
        c += (k[8] +((ub4)k[9]<<8) +((ub4)k[10]<<16) +((ub4)k[11]<<24));
        mix(a,b,c);
        k += 12; len -= 12;
    }

    /* handle the last 11 bytes */
    c += length;
    switch(len) {
        /* all the case statements fall through */
        case 11: c+=((ub4)k[10]<<24);
        case 10: c+=((ub4)k[ 9]<<16);
        case 9 : c+=((ub4)k[ 8]<< 8);
        /* the first byte of c is reserved for the length */
        case 8 : b+=((ub4)k[ 7]<<24);
        case 7 : b+=((ub4)k[ 6]<<16);
        case 6 : b+=((ub4)k[ 5]<< 8);
        case 5 : b+=k[4];
        case 4 : a+=((ub4)k[ 3]<<24);
        case 3 : a+=((ub4)k[ 2]<<16);
        case 2 : a+=((ub4)k[ 1]<< 8);
        case 1 : a+=k[0];
        /* case 0: nothing left to add */
    }
    mix(a,b,c);

#undef ub4
#undef mix

    /* report the result */
    return c;
}

/* create the hash table structure */
static lh_t *lh_create(void)
{
    lh_t *h;

    /* allocate hash structure */
    if ((h = (lh_t *)malloc(sizeof(lh_t))) == NULL)
        return NULL;

    /* allocate and clear hash table directory */
    h->h_dirsize = INITDIRSIZE;
    if ((h->h_dir = (segment_t **)malloc(h->h_dirsize * sizeof(segment_t *))) == NULL) {
        free(h);
        return NULL;
    }
    memset(h->h_dir, 0, h->h_dirsize * sizeof(segment_t *));

    /* allocate and clear first segment of hash table array */
    if ((h->h_dir[0] = (segment_t *)malloc(sizeof(segment_t))) == NULL) {
        free(h->h_dir);
        free(h);
        return NULL;
    }
    memset(h->h_dir[0], 0, sizeof(segment_t));

    /* initialize hash table control attributes */
    h->h_p     = 0;
    h->h_pmax  = SEGMENTSIZE;
    h->h_slack = SEGMENTSIZE * MAXLOADFCTR;

    return h;
}

/* expand the hash table */
static void lh_expand(lh_t *h)
{
    unsigned int pmax0;
    unsigned int newaddr;
    segment_t *seg;
    element_t **pelold;
    element_t *el, *headofold, *headofnew, *next;
    unsigned int n;

    /* calculate next new address */
    pmax0   = h->h_pmax;
    newaddr = pmax0 + h->h_p;

    /* have to enlarge directory? */
    if (h->h_dirsize <= DIRINDEX(newaddr)) {
        n = h->h_dirsize * sizeof(segment_t *);
        h->h_dirsize *= 2;
        if ((h->h_dir = (segment_t **)realloc(
                         h->h_dir, h->h_dirsize*sizeof(segment_t *))) == NULL)
             return;
        memset((char *)h->h_dir+n, 0, n);
    }

    /* have to create a new table segment? */
    if (SEGINDEX(newaddr) == 0) {
        if ((seg = (segment_t *)malloc(sizeof(segment_t))) == NULL)
            return;
        memset(seg, 0, sizeof(segment_t));
        h->h_dir[DIRINDEX(newaddr)] = seg;
    }

    /* locate P-element */
    pelold = &h->h_dir[DIRINDEX(h->h_p)]->s_element[SEGINDEX(h->h_p)];

    /* adjust the state variables */
    if (++(h->h_p) >= h->h_pmax) {
        h->h_pmax = (h->h_pmax << 1); /* == h->h_pmax * 2 */
        h->h_p = 0;
    }
    h->h_slack += MAXLOADFCTR;

    /* relocate and split between P-element new element */
    headofold = NULL;
    headofnew = NULL;
    for (el = *pelold; el != NULL; el = next) {
        next = el->e_next;
        if (el->e_hash & pmax0) {
            el->e_next = headofnew;
            headofnew  = el;
        } else {
            el->e_next = headofold;
            headofold  = el;
        }
    }
    *pelold = headofold;
    h->h_dir[DIRINDEX(newaddr)]->s_element[SEGINDEX(newaddr)] = headofnew;

    return;
}

/* shrink hash table */
static void lh_shrink(lh_t *h)
{
    segment_t *lastseg;
    element_t **pel;
    unsigned int oldlast;
    unsigned int dirsize;
    void *dir;

    /* calculate old last element */
    oldlast = h->h_p + h->h_pmax - 1;

    /* we cannot shrink below one segment */
    if (oldlast == 0)
        return;

    /* adjust the state variables */
    if (h->h_p == 0) {
        h->h_pmax = (h->h_pmax >> 1); /* == h->h_pmax / 2 */;
        h->h_p = h->h_pmax - 1;
    } else
        h->h_p--;
    h->h_slack -= MAXLOADFCTR;

    /* relocate the lost old last element to the end of the P-element */
    pel = &h->h_dir[DIRINDEX(h->h_p)]->s_element[SEGINDEX(h->h_p)];
    while (*pel != NULL)
        pel = &((*pel)->e_next);
    lastseg = h->h_dir[DIRINDEX(oldlast)];
    *pel = lastseg->s_element[SEGINDEX(oldlast)];
    lastseg->s_element[SEGINDEX(oldlast)] = NULL;

    /* if possible, deallocate the last segment */
    if (SEGINDEX(oldlast) == 0) {
        h->h_dir[DIRINDEX(oldlast)] = NULL;
        free(lastseg);
    }

    /* if possible, deallocate the end of the directory */
    if (h->h_dirsize > INITDIRSIZE && DIRINDEX(oldlast) < h->h_dirsize/2) {
        dirsize = (h->h_dirsize >> 1); /* == h->h_dirsize / 2 */
        if ((dir = (segment_t **)realloc(
                   h->h_dir, dirsize*sizeof(segment_t *))) != NULL) {
            h->h_dirsize = dirsize;
            h->h_dir = (struct segment_st **) dir;
        }
    }
    return;
}

/* insert element into hash table */
static int lh_insert(lh_t *h, const void *keyptr, int keylen, const void *datptr, int datlen, int override)
{
    unsigned int hash, addr;
    element_t *el, **pel;
    int bFound;
    void *vp;

    /* argument consistency check */
    if (h == NULL || keyptr == NULL || keylen <= 0)
        return FALSE;

    /* calculate hash address */
    hash = lh_hash((const unsigned char *)keyptr, keylen);
    addr = hash % h->h_pmax; /* unsplit region */
    if (addr < h->h_p)
        addr = hash % (h->h_pmax << 1); /* split region */

    /* locate hash element's collision list */
    pel = &h->h_dir[DIRINDEX(addr)]->s_element[SEGINDEX(addr)];

    /* check whether element is already in the hash table */
    bFound = FALSE;
    for (el = *pel; el != NULL; el = el->e_next) {
        if (   el->e_hash == hash
            && el_keylen(el) == keylen
            && memcmp(el->e_keyptr, keyptr, el_keylen(el)) == 0) {
            bFound = TRUE;
            break;
        }
    }

    /* only override on request */
    if (bFound && !override)
        return FALSE;

    /* create a duplicate of key and data */
    if ((vp = malloc(keylen+datlen)) == NULL)
        return FALSE;
    memmove(vp, datptr, datlen);
    memmove(((char *)vp)+datlen, keyptr, keylen);
    datptr = vp;
    keyptr = ((char *)vp)+datlen;

    if (bFound) {
        /* reuse existing element by freeing old contents */
        if (el->e_datptr != NULL)
            free(el->e_datptr);
    }
    else {
        /* allocate new element and chain into list */
        if ((el = (element_t *)malloc(sizeof(element_t))) == 0) {
            free(vp);
            return FALSE;
        }
        while (*pel != NULL)
            pel = &((*pel)->e_next);
        el->e_next = *pel;
        *pel = el;
    }

    /* insert contents into element structure */
    el->e_keyptr = (void *)keyptr;
    el->e_datptr = (void *)datptr;
    el->e_endptr = (char *)keyptr+keylen;
    el->e_hash   = hash;

    /* do we need to expand the table now? */
    if (--(h->h_slack) < 0)
        lh_expand(h);

    return TRUE;
}

/* lookup an element in hash table */
static int lh_lookup(lh_t *h, const void *keyptr, int keylen, void **datptr, int *datlen)
{
    unsigned int hash, addr;
    element_t *el, **pel;

    /* argument consistency check */
    if (h == NULL || keyptr == NULL || keylen <= 0)
        return FALSE;

    /* calculate hash address */
    hash = lh_hash((const unsigned char *)keyptr, keylen);
    addr = hash % h->h_pmax; /* unsplit region */
    if (addr < h->h_p)
        addr = hash % (h->h_pmax << 1); /* split region */

    /* locate hash element collision list */
    pel = &h->h_dir[DIRINDEX(addr)]->s_element[SEGINDEX(addr)];

    /* search for element in collision list */
    for (el = *pel; el != NULL; el = el->e_next) {
        if (   el->e_hash == hash
            && el_keylen(el) == keylen
            && memcmp(el->e_keyptr, keyptr, el_keylen(el)) == 0) {
            /* provide results */
            if (datptr != NULL)
                *datptr = el->e_datptr;
            if (datlen != NULL)
                *datlen = el_datlen(el);
            return TRUE;
        }
    }
    return FALSE;
}

/* delete an element in hash table */
static int lh_delete(lh_t *h, const void *keyptr, int keylen)
{
    unsigned int hash, addr;
    element_t *el, *lel, **pel;
    int rv;

    /* argument consistency check */
    if (h == NULL || keyptr == NULL || keylen <= 0)
        return FALSE;

    /* calculate hash address */
    hash = lh_hash((const unsigned char *)keyptr, keylen);
    addr = hash % h->h_pmax; /* unsplit region */
    if (addr < h->h_p)
        addr = hash % (h->h_pmax << 1); /* split region */

    /* locate hash element collision chain */
    pel = &h->h_dir[DIRINDEX(addr)]->s_element[SEGINDEX(addr)];

    /* search for element in collision chain */
    rv = FALSE;
    for (lel = NULL, el = *pel; el != NULL; lel = el, el = el->e_next) {
        if (   el->e_hash == hash
            && el_keylen(el) == keylen
            && memcmp(el->e_keyptr, keyptr, el_keylen(el)) == 0) {
            /* free key+data memory chunk */
            if (el->e_datptr != NULL)
                free(el->e_datptr);
            /* remove element from collision chain */
            if (lel == NULL)
                *pel = el->e_next;
            else
                lel->e_next = el->e_next;
            /* deallocate element structure */
            free(el);
            rv = TRUE;
            break;
        }
    }

    /* do we need to shrink the table now? */
    if (++(h->h_slack) > ((h->h_pmax + h->h_p) * (MAXLOADFCTR-MINLOADFCTR)))
        lh_shrink(h);

    return rv;
}

/* apply a callback for all elements in the hash table */
static int lh_apply(lh_t *h, lh_cb_t cb, void *ctx)
{
    element_t *el;
    unsigned int i, j;

    /* argument consistency check */
    if (h == NULL || cb == NULL)
        return FALSE;

    /* interate over all segment's entries */
    for (i = 0; i < h->h_dirsize; i++) {
        if (h->h_dir[i] == NULL)
            continue;
        /* interate over all collision chains */
        for (j = 0; j < SEGMENTSIZE; j++) {
            if (h->h_dir[i]->s_element[j] == NULL)
                continue;
            /* interate over all elements in collision chain */
            el = h->h_dir[i]->s_element[j];
            for (; el != NULL; el = el->e_next) {
                if (!cb(ctx, el->e_keyptr, el_keylen(el), el->e_datptr, el_datlen(el)))
                    return FALSE;
            }
        }
    }
    return TRUE;
}

/* destroy the whole hash table */
static int lh_destroy(lh_t *h)
{
    element_t *el, **pel, *el_next;
    unsigned int i, j;

    /* argument consistency check */
    if (h == NULL)
        return FALSE;

    /* deallocate all segment's entries */
    for (i = 0; i < h->h_dirsize; i++) {
        if (h->h_dir[i] == NULL)
            continue;
        /* deallocate all collision chains */
        for (j = 0; j < SEGMENTSIZE; j++) {
            if (h->h_dir[i]->s_element[j] == NULL)
                continue;
            /* deallocate all elements in collision chain */
            pel = &h->h_dir[i]->s_element[j];
            for (el = *pel; el != NULL; ) {
                /* deallocate key+data memory chunk */
                if (el->e_datptr != NULL)
                    free(el->e_datptr);
                el_next = el->e_next;
                free(el);
                el = el_next;
            }
        }
        free(h->h_dir[i]);
    }

    /* deallocate hash table directory */
    free(h->h_dir);

    /* deallocate hash table top-level structure */
    free(h);

    return TRUE;
}

/*
**  ____               ____
**  ____ VALUE LIBRARY ____
**
**  This part implements the actual Value library. Fortunately this
**  is now easy because it internally is just based on the above
**  full-featured linear hashing library.
*/

/*
 * usually val_object_t.data is a pointer val_object_t.data.p,
 * but VAL_INLINE signals val_object_t.data is actual data
 * val_object_t.data.[csilfd].
 */
enum {
    VAL_INLINE = 1<<31
};

/* the internal representation of a value object */
typedef struct {
    int type;
    union {
        val_t *v;
        void  *p;
        char   c;
        short  s;
        int    i;
        long   l;
        float  f;
        double d;
    } data;
    char *desc;
} val_object_t;

/* the val_t internally is just a hash table */
struct val_s {
    lh_t *lh;
};

/* determine address of an object's storage */
static void *val_storage(val_object_t *obj)
{
    void *storage;

    /* argument consistency check */
    if (obj == NULL)
        return NULL;

    /* address determination */
    if (obj->type & VAL_INLINE) {
        switch (obj->type & ~VAL_INLINE) {
            case VAL_TYPE_VAL:    storage = &obj->data.v; break;
            case VAL_TYPE_PTR:    storage = &obj->data.p; break;
            case VAL_TYPE_CHAR:   storage = &obj->data.c; break;
            case VAL_TYPE_SHORT:  storage = &obj->data.s; break;
            case VAL_TYPE_INT:    storage = &obj->data.i; break;
            case VAL_TYPE_LONG:   storage = &obj->data.l; break;
            case VAL_TYPE_FLOAT:  storage = &obj->data.f; break;
            case VAL_TYPE_DOUBLE: storage = &obj->data.d; break;
            default:              storage = NULL; break; /* cannot happen */
        }
    }
    else
        storage = obj->data.p;

    return storage;
}

/* create object */
val_rc_t val_create(val_t **pval)
{
    val_t *val;

    /* argument consistency check */
    if (pval == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* create top-level structure */
    if ((val = (val_t *)malloc(sizeof(val_t))) == NULL)
        return VAL_RC(VAL_ERR_SYS);

    /* create hash table */
    if ((val->lh = lh_create()) == NULL) {
        free(val);
        return VAL_RC(VAL_ERR_SYS);
    }

    /* pass result to caller */
    *pval = val;

    return VAL_OK;
}

/* internal lh_apply() callback for use with val_destroy() */
static int val_destroy_cb(void *_ctx,
                          const void *keyptr, int keylen,
                          const void *datptr, int datlen)
{
    val_object_t *obj;

    /* free description string */
    if ((obj = (val_object_t *)datptr) != NULL)
        if (obj->desc != NULL)
            free(obj->desc);

    return TRUE;
}

/* destroy object */
val_rc_t val_destroy(val_t *val)
{
    /* argument consistency check */
    if (val == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* destroy all hash table elements */
    lh_apply(val->lh, val_destroy_cb, NULL);

    /* destroy hash table */
    if (!lh_destroy(val->lh))
        return VAL_RC(VAL_ERR_SYS);

    /* destroy top-level structure */
    free(val);

    return VAL_OK;
}

/* register a value */
val_rc_t val_reg(val_t *val, const char *name, int type, const char *desc, void *storage)
{
    val_object_t *obj;
    val_object_t newobj;
    const char *cp;
    val_t *child;

    /* argument consistency check */
    if (val == NULL || name == NULL || type == 0)
        return VAL_RC(VAL_ERR_ARG);

    /* recursive step-down on structured name */
    if ((cp = strchr(name, '.')) != NULL) {
        if (!lh_lookup(val->lh, name, cp-name, (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if (!(obj->type & VAL_TYPE_VAL))
            return VAL_RC(VAL_ERR_USE);
        child = *(val_t **)(val_storage(obj));
        return val_reg(child, cp+1, type, desc, storage);
    }

    /* create a new value object */
    if (desc != NULL)
        newobj.desc = strdup(desc);
    else
        newobj.desc = NULL;
    if (storage == NULL) {
        newobj.type   = (type | VAL_INLINE);
        newobj.data.l = 0;
    }
    else {
        newobj.type   = (type & ~VAL_INLINE);
        newobj.data.p = storage;
    }

    /* insert value into hash table */
    if (!lh_insert(val->lh, name, strlen(name), &newobj, sizeof(newobj), 1))
        return VAL_RC(VAL_ERR_HSH);

    return VAL_OK;
}

val_rc_t val_unreg(val_t *val, const char *name)
{
    val_object_t *obj;
    const char *cp;
    val_t *child;

    /* argument consistency check */
    if (val == NULL || name == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* recursive step-down on structured name */
    if ((cp = strchr(name, '.')) != NULL) {
        if (!lh_lookup(val->lh, name, cp-name, (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if (!(obj->type & VAL_TYPE_VAL))
            return VAL_RC(VAL_ERR_USE);
        child = *(val_t **)(val_storage(obj));
        return val_unreg(child, cp+1);
    }

    /* try to lookup object in hash table */
    if (!lh_lookup(val->lh, name, strlen(name), (void **)(void *)&obj, NULL))
        return VAL_RC(VAL_ERR_ARG);

    /* destroy value object */
    if (obj->desc != NULL)
        free(obj->desc);

    /* delete value from hash table */
    if (!lh_delete(val->lh, name, strlen(name)))
        return VAL_RC(VAL_ERR_HSH);

    return VAL_OK;
}

/* query information about a value */
val_rc_t val_query(val_t *val, const char *name,
                   int *ptype, char **pdesc, void **pstorage)
{
    char *cp;
    val_object_t *obj;
    val_t *child;

    /* argument consistency check */
    if (val == NULL || name == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* recursive step-down on structured name */
    if ((cp = strchr(name, '.')) != NULL) {
        if (!lh_lookup(val->lh, name, cp-name, (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if (!(obj->type & VAL_TYPE_VAL))
            return VAL_RC(VAL_ERR_USE);
        child = *(val_t **)(val_storage(obj));
        return val_query(child, cp+1, ptype, pdesc, pstorage);
    }

    /* try to lookup object in hash table */
    if (!lh_lookup(val->lh, name, strlen(name), (void **)(void *)&obj, NULL))
        return VAL_RC(VAL_ERR_ARG);

    /* pass queried information to caller */
    if (ptype != NULL)
        *ptype = (obj->type & ~VAL_INLINE);
    if (pdesc != NULL)
        *pdesc = obj->desc;
    if (pstorage != NULL)
        *pstorage = val_storage(obj);

    return VAL_OK;
}

/* set a value (va_list variant) */
val_rc_t val_vset(val_t *val, const char *name, va_list ap)
{
    val_object_t *obj;
    void *storage;
    const char *cp;
    val_t *child;

    /* argument consistency check */
    if (val == NULL || name == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* recursive step-down on structured name */
    if ((cp = strchr(name, '.')) != NULL) {
        if (!lh_lookup(val->lh, name, cp-name, (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if (!(obj->type & VAL_TYPE_VAL))
            return VAL_RC(VAL_ERR_USE);
        child = *(val_t **)(val_storage(obj));
        return val_vset(child, cp+1, ap);
    }

    /* try to lookup object in hash table */
    if (!lh_lookup(val->lh, name, strlen(name), (void **)(void *)&obj, NULL))
        return VAL_RC(VAL_ERR_ARG);

    /* determine value storage */
    if ((storage = val_storage(obj)) == NULL)
        return VAL_RC(VAL_ERR_INT);

    /* copy value from variable argument into storage location */
    switch (obj->type & ~VAL_INLINE) {
        case VAL_TYPE_VAL:    *(val_t **)storage = (val_t *)va_arg(ap, void *); break;
        case VAL_TYPE_PTR:    *(char  **)storage = (char  *)va_arg(ap, void *); break;
        case VAL_TYPE_CHAR:   *(char   *)storage = (char   )va_arg(ap, int   ); break;
        case VAL_TYPE_SHORT:  *(short  *)storage = (short  )va_arg(ap, int   ); break;
        case VAL_TYPE_INT:    *(int    *)storage = (int    )va_arg(ap, int   ); break;
        case VAL_TYPE_LONG:   *(long   *)storage = (long   )va_arg(ap, long  ); break;
        case VAL_TYPE_FLOAT:  *(float  *)storage = (float  )va_arg(ap, double); break;
        case VAL_TYPE_DOUBLE: *(double *)storage = (double )va_arg(ap, double); break;
        default: break; /* cannot happen */
    }

    return VAL_OK;
}

/* set a value */
val_rc_t val_set(val_t *val, const char *name, ...)
{
    val_rc_t rc;
    va_list ap;

    /* argument consistency check */
    if (val == NULL || name == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* just pass-through to va_list variant */
    va_start(ap, name);
    rc = val_vset(val, name, ap);
    va_end(ap);

    return VAL_RC(rc);
}

/* get a value (va_list variant) */
val_rc_t val_vget(val_t *val, const char *name, va_list ap)
{
    val_object_t *obj;
    void *storage;
    const char *cp;
    val_t *child;

    /* argument consistency check */
    if (val == NULL || name == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* recursive step-down on structured name */
    if ((cp = strchr(name, '.')) != NULL) {
        if (!lh_lookup(val->lh, name, cp-name, (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if (!(obj->type & VAL_TYPE_VAL))
            return VAL_RC(VAL_ERR_USE);
        child = *(val_t **)(val_storage(obj));
        return val_vget(child, cp+1, ap);
    }

    /* try to lookup object in hash table */
    if (!lh_lookup(val->lh, name, strlen(name), (void **)(void *)&obj, NULL))
        return VAL_RC(VAL_ERR_ARG);

    /* determine value storage */
    if ((storage = val_storage(obj)) == NULL)
        return VAL_RC(VAL_ERR_INT);

    /* copy value from storage location into variable argument pointer location */
    switch (obj->type & ~VAL_INLINE) {
        case VAL_TYPE_VAL:    *((val_t **)va_arg(ap, void   *)) = *(val_t **)storage; break;
        case VAL_TYPE_PTR:    *((char  **)va_arg(ap, void   *)) = *(char  **)storage; break;
        case VAL_TYPE_CHAR:   *((char   *)va_arg(ap, int    *)) = *(char   *)storage; break;
        case VAL_TYPE_SHORT:  *((short  *)va_arg(ap, int    *)) = *(short  *)storage; break;
        case VAL_TYPE_INT:    *((int    *)va_arg(ap, int    *)) = *(int    *)storage; break;
        case VAL_TYPE_LONG:   *((long   *)va_arg(ap, long   *)) = *(long   *)storage; break;
        case VAL_TYPE_FLOAT:  *((float  *)va_arg(ap, double *)) = *(float  *)storage; break;
        case VAL_TYPE_DOUBLE: *((double *)va_arg(ap, double *)) = *(double *)storage; break;
        default: break; /* cannot happen */
    }

    return VAL_OK;
}

/* get a value */
val_rc_t val_get(val_t *val, const char *name, ...)
{
    val_rc_t rc;
    va_list ap;

    /* argument consistency check */
    if (val == NULL || name == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* just pass-through to va_list variant */
    va_start(ap, name);
    rc = val_vget(val, name, ap);
    va_end(ap);

    return VAL_RC(rc);
}

/* internal lh_apply() recursion callback context structure */
typedef struct {
    val_t    *val;
    char     *name;
    int       prefixlen;
    int       depth;
    val_cb_t  cb;
    void     *ctx;
    val_rc_t  rc;
} val_apply_ctx_t;

/* forward declaration */
static val_rc_t val_apply_internal(val_t *, const char *, int, int, val_cb_t, void *);

/* internal lh_apply() recursion callback for use with val_apply() */
static int (val_apply_cb)(void *_ctx, const void *keyptr, int keylen, const void *datptr, int datlen)
{
    val_apply_ctx_t *ctx = (val_apply_ctx_t *)_ctx;
    char name[VAL_MAXNAME+1];
    int prefixlen;

    /* on-the-fly create NUL-terminated concatenated name string */
    if ((strlen(ctx->name) + 1 + keylen) > VAL_MAXNAME) {
        ctx->rc = VAL_ERR_MEM;
        return FALSE;
    }
    if (strlen(ctx->name) > 0) {
        strcpy(name, ctx->name);
        strcat(name, ".");
        prefixlen = ctx->prefixlen + 1;
    }
    else {
        *name = '\0';
        prefixlen = ctx->prefixlen;
    }
    strncat(name, (char *)keyptr, keylen);

    /* recursive step-down */
    if ((ctx->rc = val_apply_internal(ctx->val, name, prefixlen,
                                      ctx->depth, ctx->cb, ctx->ctx)) != VAL_OK)
        return FALSE;

    return TRUE;
}

/* internal API-increased variant of val_apply() */
static val_rc_t val_apply_internal(val_t *val, const char *name, int prefixlen,
                                   int depth, val_cb_t cb, void *ctx)
{
    val_object_t *obj;
    val_t *child;
    char *cp;
    val_rc_t rc;
    val_apply_ctx_t val_ctx;

    if (name[prefixlen] == '\0') {
        /* CASE 1: apply to all elements
           prefix="foo.bar.", remainder="" */
        val_ctx.val       = val;
        val_ctx.name      = (char *)name;
        val_ctx.prefixlen = prefixlen;
        val_ctx.depth     = depth;
        val_ctx.cb        = cb;
        val_ctx.ctx       = ctx;
        val_ctx.rc        = VAL_OK;
        if (!lh_apply(val->lh, val_apply_cb, &val_ctx))
            return VAL_RC(VAL_ERR_SYS);
    }
    else if ((cp = strchr(name+prefixlen, '.')) != NULL) {
        /* CASE 2: still stepping-down for structured name
           prefix="foo.bar.", remainder="quux.baz" */
        if (!lh_lookup(val->lh, name+prefixlen, cp-(name+prefixlen), (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if (!(obj->type & VAL_TYPE_VAL))
            return VAL_RC(VAL_ERR_USE);
        child = *(val_t **)(val_storage(obj));
        if (depth == 0)
            return VAL_OK;
        return val_apply_internal(child, name, cp-name+1, depth-1, cb, ctx);
    } else {
        /* CASE 3: reached last component of structured name
           prefix="foo.bar.quux.", remainder="baz" */
        if (!lh_lookup(val->lh, name+prefixlen, strlen(name+prefixlen), (void **)(void *)&obj, NULL))
            return VAL_RC(VAL_ERR_ARG);
        if ((rc = cb(ctx, name, (obj->type & ~VAL_INLINE),
                     obj->desc, val_storage(obj))) != VAL_OK)
            return VAL_RC(rc);
        if (obj->type & VAL_TYPE_VAL) {
            if (depth == 0)
                return VAL_OK;
            child = *(val_t **)(val_storage(obj));
            return val_apply_internal(child, name, strlen(name), depth-1, cb, ctx);
        }
    }
    return VAL_OK;
}

/* apply a callback to each value */
val_rc_t val_apply(val_t *val, const char *name, int depth, val_cb_t cb, void *ctx)
{
    /* argument consistency check */
    if (val == NULL || name == NULL || depth < 0 || cb == NULL)
        return VAL_RC(VAL_ERR_ARG);

    /* just pass-through to internal API-extended variant */
    return val_apply_internal(val, name, 0, depth, cb, ctx);
}

