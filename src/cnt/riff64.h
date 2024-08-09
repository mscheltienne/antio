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

#ifndef RIFF64_H
#define RIFF64_H

#include <stdio.h>

#include <cnt/riff.h>

typedef chunk_t chunk64_t;

/*
  check 'RIFF' tag, formtype is set on return
  caller is responsible for readable stream
*/
int riff64_form_open(FILE *f, chunk64_t *chunk, fourcc_t *formtype);

/*
  scan riff for specified chunk in a valid list or form parent
  no recursion is performed,
  you have to look active for each chunk you want to read (climb
  the tree by subsequent calls to riff_open, riff_list_open)
*/
int riff64_list_open(FILE *f, chunk64_t *chunk, fourcc_t listtype, chunk64_t parent);
int riff64_open(FILE *f, chunk64_t *chunk, fourcc_t id, chunk64_t parent);

/*
  unspecific search; returns the riff id (and listid for <LIST>) of a subchunk;
  you can call this until RIFF_NOCHUNK is returned
*/
int riff64_fetch(FILE *f, chunk64_t *chunk, fourcc_t *listid, chunk64_t parent, int child);

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
int riff64_form_new(FILE *f, chunk64_t *chunk, fourcc_t formtype);
int riff64_list_new(FILE *f, chunk64_t *chunk, fourcc_t listtype, chunk64_t *parent);

int riff64_new(FILE *f, chunk64_t *chunk, fourcc_t chunktype, chunk64_t *parent);
/*
  write the size of the chunk into file, if dumptree is nonzero the
  parents data are written too,
  last riff_close must set dumptree
  dumptree always if you don't know which one is the last
*/
int riff64_close(FILE *f, chunk64_t chunk);

int riff64_put_chunk(FILE *f, chunk64_t out);

/*
  random access functions
  return: RIFFERR_...  status

  riff_write maintains the chunk size (only sequential write allowed!)
  riff_seek works like fseek with "whence" in the chunk data area
  (supported only for read access!)
*/
int riff64_write(const char *buf, size_t size, size_t num_items, FILE *f, chunk64_t *chunk);
int riff64_read(char *buf, size_t size, size_t num_items, FILE *f, chunk64_t chunk);
int riff64_seek(FILE *f, uint64_t offset, int whence, chunk64_t chunk);

uint64_t riff64_get_chunk_size(chunk64_t chunk);
fourcc_t riff64_get_chunk_id(chunk64_t chunk);

#endif
