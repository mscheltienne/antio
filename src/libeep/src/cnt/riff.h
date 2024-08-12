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

#ifndef RIFF_H
#define RIFF_H
#define RCS_RIFF_H "$RCSfile: riff.h,v $ $Revision: 1762 $"

#include <stdio.h>
#include <eep/stdint.h>

/* RIFF access return values */
#define RIFFERR_NONE 0
#define RIFFERR_FILE 1     /* fread/fwrite failed */
#define RIFFERR_NOCHUNK 2  /* expected chunk not found */
#define RIFFERR_TREE 3     /* inconsistencies in riff data */

/* 32 bit unsigned type for FOURCC's */
typedef unsigned int fourcc_t;

/* converts 4 8bit digits to fourcc_t */
#define FOURCC(a,b,c,d) (  ((fourcc_t) a)       | ((fourcc_t) b <<  8) | \
                           ((fourcc_t) c) << 16 | ((fourcc_t) d << 24)     )


/*
  chunk information structure,
  caller need no direct access to its members, but is resposible for unique
  instances  of chunk_t for all chunks in riff
*/
struct chunk {
  fourcc_t      id;
  uint64_t      start;
  uint64_t      size;
  struct chunk  *parent;
};
typedef struct chunk chunk_t;

/* the standard FOURCC tags and formats known by the library*/
#define FOURCC_RIFF FOURCC('R', 'I', 'F', 'F')
#define FOURCC_RF64 FOURCC('R', 'F', '6', '4')
#define FOURCC_LIST FOURCC('L', 'I', 'S', 'T')

/*
  check 'RIFF' tag, formtype is set on return
  caller is responsible for readable stream
*/
int riff_form_open(FILE *f, chunk_t *chunk, fourcc_t *formtype);

/*
  scan riff for specified chunk in a valid list or form parent
  no recursion is performed,
  you have to look active for each chunk you want to read (climb
  the tree by subsequent calls to riff_open, riff_list_open)
*/
int riff_list_open(FILE *f, chunk_t *chunk, fourcc_t listtype, chunk_t parent);
int riff_open(FILE *f, chunk_t *chunk, fourcc_t id, chunk_t parent);

/*
  unspecific search; returns the riff id (and listid for <LIST>) of a subchunk;
  you can call this until RIFF_NOCHUNK is returned
*/
int riff_fetch(FILE *f, chunk_t *chunk, fourcc_t *listid,
               chunk_t parent, int child);

/*
  creating new chunks:

  you have to create a form ('RIFF' chunk) first by definition
  other chunks has to be filled with data by riff_write and to be closed by
  riff_close before starting next!

  create a riff tree as follows:

  fopen

  riff_form_new
    riff_new; riff_write; riff_close
    riff_new; riff_write; riff_close
    ...
    riff_list_new;
      riff_list_new
        riff_new; riff_write; riff_close
      riff_new; riff_write; riff_close
    riff_close
  riff_close

  fclose


NOTE: It seems riff_new and other functions do not longer position themselves at the 
      end of the file. Before calling riff_new the filepointer should point to the end
      of the file with "fseek(f, 0, SEEK_END)", or it may possible overwrite previous
      chunks. I am not sure why this is done, but there probably is some reason. -- jwiskerke
*/
int riff_form_new(FILE *f, chunk_t *chunk, fourcc_t formtype);
int riff_list_new(FILE *f, chunk_t *chunk, fourcc_t listtype, chunk_t *parent);

int riff_new(FILE *f, chunk_t *chunk, fourcc_t chunktype, chunk_t *parent);
/*
  write the size of the chunk into file, if dumptree is nonzero the
  parents data are written too,
  last riff_close must set dumptree
  dumptree always if you don't know which one is the last
*/
int riff_close(FILE *f, chunk_t chunk);

int riff_put_chunk(FILE *f, chunk_t out);

/*
  random access functions
  return: RIFFERR_...  status

  riff_write maintains the chunk size (only sequential write allowed!)
  riff_seek works like fseek with "whence" in the chunk data area
  (supported only for read access!)
*/
int riff_write(const char *buf, size_t size, size_t num_items, FILE *f, chunk_t *chunk);
int riff_read(char *buf, size_t size, size_t num_items, FILE *f, chunk_t chunk);
int riff_seek(FILE *f, long offset, int whence, chunk_t chunk);

long     riff_get_chunk_size(chunk_t chunk);
fourcc_t riff_get_chunk_id(chunk_t chunk);

#endif
