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

#include <eep/inttypes.h>

#include <cnt/riff64.h>
#include <eep/eepio.h>
#include <eep/eepraw.h>

#ifndef SEEK_SET
# define SEEK_SET 0
#endif
#ifndef SEEK_CUR
# define SEEK_CUR 1
#endif
#ifndef SEEK_END
# define SEEK_END 2
#endif

#define CHUNK64HEADER_SIZE 12
#define PARENT64HEADER_SIZE 16

// #define RIFF_DEBUG

void
_riff64_dump_chunk(const chunk64_t *c) {
  fprintf(stderr, "chunk(");
  fprintf(stderr, " fcc=%c%c%c%c", ((char *)&c->id)[0], ((char *)&c->id)[1], ((char *)&c->id)[2], ((char *)&c->id)[3]);
  fprintf(stderr, ", start=%10" PRIu64, c->start);
  fprintf(stderr, ", size=%10" PRIu64, c->size);
  if(c->size % 4 != 0) {
    fprintf(stderr, " size not multiple of 4!");
  }
  fprintf(stderr, ")\n");
}

int _riff64_get_id(FILE *f, fourcc_t *in) {
  char id[4];

  eepio_fread(id, 4, 1, f);
  *in = FOURCC(id[0], id[1], id[2], id[3]);

  return ferror(f);
}

int _riff64_put_id(FILE *f, fourcc_t out) {
  char id[4];

  id[0] =  (char) (out & 0xff);
  id[1] =  (char) ((out >> 8) & 0xff);
  id[2] =  (char) ((out >> 16) & 0xff);
  id[3] =  (char) ((out >> 24) & 0xff);

  eepio_fwrite(id, 4, 1, f);

  return ferror(f);
}

int _riff64_get_chunk(FILE *f, chunk64_t *in) {
  in->start = eepio_ftell(f);
  _riff64_get_id(f, &(in->id));
  read_u64(f, &(in->size));
#ifdef RIFF_DEBUG
  fprintf(stderr, "%s(%i): ", __FUNCTION__, __LINE__);
  _riff64_dump_chunk(in);
#endif
  return ferror(f);
}

int riff64_put_chunk(FILE *f, chunk64_t out) {
  _riff64_put_id(f, out.id);
  write_u64(f, out.size);
#ifdef RIFF_DEBUG
  fprintf(stderr, "%s(%i): ", __FUNCTION__, __LINE__);
  _riff64_dump_chunk(&out);
#endif
  return ferror(f);
}

int riff64_form_open(FILE *f, chunk64_t *chunk, fourcc_t *formtype) {
  rewind(f);

  chunk->parent = NULL;
  _riff64_get_chunk(f, chunk);
  if (chunk->id == FOURCC_RF64) {
    _riff64_get_id(f, formtype);
    return RIFFERR_NONE;
  }
  else {
    return RIFFERR_NOCHUNK;
  }
}

int riff64_list_open(FILE *f, chunk64_t *chunk, fourcc_t listtype, chunk64_t parent) {
  fourcc_t curlisttype;
  char match = 0;
  uint64_t nextchunk = 0;
  uint64_t skipsize = 0;

  /* locate the start of our tree level (the parents data area) */

  eepio_fseek(f, parent.start + PARENT64HEADER_SIZE, SEEK_SET);
  do {
    eepio_fseek(f, nextchunk, SEEK_CUR);
    if (_riff64_get_chunk(f, chunk)) return RIFFERR_FILE;
    if (chunk->id == FOURCC_LIST) {
      _riff64_get_id(f, &curlisttype);
      if (curlisttype == listtype) {
        match = 1;
      }
      else {
        skipsize += chunk->size + CHUNK64HEADER_SIZE + (chunk->size & 0x01);
        nextchunk = chunk->size - 4 + (chunk->size & 0x01);
      }
    }
    else {
      skipsize += chunk->size + CHUNK64HEADER_SIZE + (chunk->size & 0x01);
      nextchunk = chunk->size + (chunk->size & 0x01);
    }
  } while (!match && skipsize < parent.size - 1);

  if (match)
    return RIFFERR_NONE;
  else
    return RIFFERR_NOCHUNK;

}

int riff64_open(FILE *f, chunk64_t *chunk, fourcc_t id, chunk64_t parent) {
  char match = 0;
  uint64_t nextchunk = 0;
  uint64_t skipsize = 0;

  /* go to parent data area */
  eepio_fseek(f, parent.start + PARENT64HEADER_SIZE, SEEK_SET);

  /* loop true the childs on this level, no recursion into tree! */
  do {
    eepio_fseek(f, nextchunk, SEEK_CUR);
    if (_riff64_get_chunk(f, chunk)) return RIFFERR_FILE;
    if (chunk->id == id) {
      match = 1;
    }
    else {
      skipsize += chunk->size + CHUNK64HEADER_SIZE + (chunk->size & 0x01);
      nextchunk = chunk->size + (chunk->size & 0x01);
    }
  } while (!match && skipsize < parent.size);

  if (match) {
    return RIFFERR_NONE;
  } else {
    return RIFFERR_NOCHUNK;
  }

}

int riff64_fetch(FILE *f, chunk64_t *chunk, fourcc_t *listid, chunk64_t parent, int child) {
  int s, i = 0;
  uint64_t got = 0;

  /* locate parent data area start */
  eepio_fseek(f, parent.start + PARENT64HEADER_SIZE, SEEK_SET);

  s = _riff64_get_chunk(f, chunk);
  while (!s && i != child && got + chunk->size < parent.size)
  {
    eepio_fseek(f, chunk->size + (chunk->size & 1), SEEK_CUR);
    got += CHUNK64HEADER_SIZE + chunk->size + (chunk->size & 1);
    s = _riff64_get_chunk(f, chunk);
    i++;
  }

  if (s || got + chunk->size > parent.size) {
    return RIFFERR_NOCHUNK;
  }
  else {
    if (chunk->id == FOURCC_LIST)
      _riff64_get_id(f, listid);
    return RIFFERR_NONE;
  }
}

int riff64_form_new(FILE *f, chunk64_t *chunk, fourcc_t formtype) {
  rewind(f);

  chunk->id = FOURCC_RF64;
  chunk->parent = NULL;
  chunk->start = 0;
  chunk->size = 4;

  if (riff64_put_chunk(f, *chunk)) return RIFFERR_FILE;
  if (_riff64_put_id(f, formtype)) return RIFFERR_FILE;

  return RIFFERR_NONE;
}


int riff64_list_new(FILE *f, chunk64_t *chunk, fourcc_t listtype, chunk64_t *parent) {
  chunk64_t *x;

  chunk->id = FOURCC_LIST;
  chunk->start = eepio_ftell(f);
  chunk->size = 4;
  chunk->parent = parent;


  if (riff64_put_chunk(f, *chunk)) return RIFFERR_FILE;
  if (_riff64_put_id(f, listtype)) return RIFFERR_FILE;

  x = chunk;
  while (x->parent != NULL) {
    x = x->parent;
    x->size += PARENT64HEADER_SIZE;
  }

  return RIFFERR_NONE;
}


int riff64_new(FILE *f, chunk64_t *chunk, fourcc_t chunktype, chunk64_t *parent) {
  chunk64_t *x;
  chunk->id = chunktype;
  chunk->start = eepio_ftell(f);
  chunk->parent = parent;
  chunk->size = 0;

  if (riff64_put_chunk(f, *chunk)) return RIFFERR_FILE;
  x = chunk;
  while (x->parent != NULL) {
    x = x->parent;
    x->size += CHUNK64HEADER_SIZE;
  }

  return ferror(f);
}

int riff64_close(FILE *f, chunk64_t chunk) {
  uint64_t fillbytes;
  uint64_t start;
  chunk64_t *x;
  char junk = '\0';
#ifdef RIFF_DEBUG
  fprintf(stderr, "%s(%i): ", __FUNCTION__, __LINE__);
  _riff64_dump_chunk(&chunk);
#endif
  /*eepio_fseek(f, 0, SEEK_END);*/
  start = eepio_ftell(f);
  fillbytes = start & 0x01;
#ifdef RIFF_DEBUG
  fprintf(stderr, "%s(%i): ", __FUNCTION__, __LINE__);
  _riff64_dump_chunk(&chunk);
#endif
  /* write the chunk header */
  eepio_fseek(f, chunk.start, SEEK_SET);
  if (riff64_put_chunk(f, chunk)) return RIFFERR_FILE;

  /* tell the parents about their new size */
  x = &chunk;
  while (x->parent != NULL) {
    x = x->parent;
    x->size += fillbytes + chunk.size;
    eepio_fseek(f, x->start, SEEK_SET);
    if (riff64_put_chunk(f, *x)) return RIFFERR_FILE;
  }

  /* force next start at even filepos */
  eepio_fseek(f, start, SEEK_SET);
  if (fillbytes) eepio_fwrite(&junk, 1, 1, f);

  return RIFFERR_NONE;
}

int riff64_write(const char *buf, size_t size, size_t num_items, FILE *f, chunk64_t *chunk) {
  uint64_t sizeinc = size * num_items;

  if (eepio_fwrite(buf, size, num_items, f) != num_items) return RIFFERR_FILE;
  chunk->size += sizeinc;

  return RIFFERR_NONE;
}

int riff64_read(char *buf, size_t size, size_t num_items,
                    FILE *f, chunk64_t chunk)
{
  if (eepio_fread(buf, size, num_items, f) != num_items) return RIFFERR_FILE;

  return RIFFERR_NONE;
}

int riff64_seek(FILE *f, uint64_t offset, int whence, chunk64_t chunk) {
  uint64_t effpos=0;

  switch (whence) {
    case SEEK_SET: effpos = chunk.start + CHUNK64HEADER_SIZE + offset;
                   break;
    case SEEK_CUR: effpos = offset;
                   break;
    case SEEK_END: effpos = chunk.start + CHUNK64HEADER_SIZE + chunk.size;
                   break;
  }
  if (eepio_fseek(f, effpos, (whence != SEEK_CUR) ? SEEK_SET : SEEK_CUR))
    return RIFFERR_FILE;
  else
    return RIFFERR_NONE;
}

uint64_t riff64_get_chunk_size(chunk64_t chunk) {
  return chunk.size;
}

fourcc_t riff64_get_chunk_id(chunk64_t chunk) {
  return chunk.id;
}
