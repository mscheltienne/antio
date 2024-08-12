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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <cnt/cnt_private.h>
#include <cnt/raw3.h>

#ifdef COMPILE_RCS
char RCS_raw3_h[] = RCS_RAW3_H;
char RCS_raw3_c[] = "$RCSfile: raw3.c,v $ $Revision: 2415 $";
#endif

/* #ifdef RAW3_CHECK obsolete */
/* indicator flags for the citical compression methods */
short ERR_FLAG_16 = 0;
short ERR_FLAG_0  = 0;
short ERR_FLAG_EPOCH = 0;
/* #endif */

void    raw3_set_ERR_FLAG_16(short n) { ERR_FLAG_16 = n; }
short   raw3_get_ERR_FLAG_16()        { return ERR_FLAG_16; }
void    raw3_set_ERR_FLAG_0(short n) { ERR_FLAG_0 = n; }
short   raw3_get_ERR_FLAG_0()        { return ERR_FLAG_0; }
void    raw3_set_ERR_FLAG_EPOCH(short n) { ERR_FLAG_EPOCH = n; }
short   raw3_get_ERR_FLAG_EPOCH()        { return ERR_FLAG_EPOCH; }

unsigned char CheckVerbose = 0;
void raw3_setVerbose(int onoff)
{
    assert(onoff == 0 || onoff == 1);
    CheckVerbose = (unsigned char) onoff;
}

/*
  known residual prediction/encoding methods
  these codes are stored in data files - never change this!
*/

#define RAW3_COPY  0  /* no residuals, original 16-bit values stored */
#define RAW3_TIME  1  /* 16 bit residuals from first deviation       */
#define RAW3_TIME2 2  /* 16 bit residuals from second deviation      */
#define RAW3_CHAN  3  /* 16 bit residuals from difference of first deviation */
                      /* and first dev. of neighbor channel   */

#define RAW3_COPY_32   8  /* the same for 32 bit sample data */
#define RAW3_TIME_32   9
#define RAW3_TIME2_32 10
#define RAW3_CHAN_32  11

/*
  find the number of bits needed to store the passed (signed!) value
*/

int bitc(sraw_t x)
{
  static int nbits[128] = {
/*  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 */

    1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,

    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
  };

  register int y = x;

  /* positive number which needs the same number of bits */
  if (y < 0) y = -(y+1);

  if (y < 0x00000080)
    return nbits[y];
  else if (y < 0x00008000)
    return nbits[y >> 8] + 8;
  else if (y < 0x00800000)
    return nbits[y >> 16] + 16;
  else
    return  nbits[y >> 24] + 24;
}

int huffman_size(int *bitfreq, int n, int *nbits, int *nexcbits)
{
  int lmin = 1000000000;
  int lcur;
  int  i, j, nexc;

  /* don't store with one bit per sample point */
  bitfreq[2] += bitfreq[1]; bitfreq[1] = 0;

  /* determine the max bits to store */
  j = 32;
  while(bitfreq[j] == 0) j--;
  *nexcbits = j;

  /*
  calculate the length needed for coding in i bits with exceptions for all
  the residuals out of range
  */
  for (i = 2; i <= *nexcbits; i++) {

    /* all residuals out of bitrange are exceptions */
    nexc = 0;
    for (j = i + 1; j <= *nexcbits; j++) {
      nexc += bitfreq[j];
    }

    /*
    we need one value in bitrange for exception coding, which will cause
    some more exceptions
    e.g. for 4-bit it's every 16th value; we divide it because the lower edge value
    we are going to use for this code is less frequent than the centers
    */

    nexc += (n / (1 << i)) / 2;

    lcur = (n - 1) * i + nexc * (*nexcbits);
    /*
    lcur = (n - nexc) * i + nexc * (*nexcbits + i);
            |                 |
            |                 exceptions (code + full value)
            fitting values
    */

    /* note the best i for return */
    if (lcur < lmin) {
      lmin = lcur;
      *nbits = i;
    }
  }

  /* bits 2 bytes */
  lcur = lmin / 8;
  if (lmin % 8)
    lcur++;

  /*
  printf("%d\t%d\n", *nbits, *nexcbits);
  */

  return lcur;
}

/* --------------------------------------------------------------------
  raw3 compressed data vectors

  TIME/TIME2/CHAN:| method | nbits | nexcbits | x[0] | x[1] .. x[n-1]
                      4        4         4       16    nbits or nbits + nexcbits

  ..._32         :| method | nbits | nexcbits | x[0] | x[1] .. x[n-1]
                      4        6         6       32    nbits or nbits + nexcbits

  COPY:           | method | dummy | x[0] .. x[n-1]
                      4        4         16

  COPY_32:        | method | dummy | x[0] .. x[n-1]
                      4        4         32

*/

int huffman(sraw_t *in, int n, int method, int nbits, int nexcbits,
              unsigned char *out)
{
  int nout = 0;
  int nin;
  int bitin, bitout;
  sraw_t inval, loval, hival;
  char outval;
  int exc, check_exc = nbits != nexcbits;
  int nbits_1 = nbits - 1;
  int nexcbits_1 = nexcbits - 1;
  /*int i,j; */
  /*static int xxxx = 0; */

  static unsigned char setoutbit[8] = {
    0x01, 0x02, 0x04, 0x08,
    0x10, 0x20, 0x40, 0x80
  };

  static unsigned int setinbit[32] = {
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000
  };

  /* first 4 bits for the method */
  out[0] = method << 4;
 /*
  printf("method: %d, %d, %d\n", method, out[0]>>4, out[0]);
  fflush(stdout);
 */
  /* the rest depends on the method */
  if (method != RAW3_COPY && method != RAW3_COPY_32) {

    /* 32 bit versions (6 bits to store number of bits, 32 bit start val.) */
    if (method & 0x08) {
      out[0] |= nbits >> 2;
      out[1] = (nbits << 6) | nexcbits;
      out[2] = in[0] >> 24;
      out[3] = in[0] >> 16;
      out[4] = in[0] >>  8;
      out[5] = in[0];
      outval = 0;
      nout = 6;
      bitout = 7;
    }

    /* 16 bit versions (4 bits to store number of bits, 16 bit start val.) */
    else {
      out[0] |= nbits;
      out[1] = (nexcbits << 4) | ((in[0] >> 12) & 0x0f);
      out[2] = (in[0] >> 4) & 0xff;
      outval = (in[0] & 0x0f) << 4;
      nout = 3;
      bitout = 3;
    }

    hival = 1 << nbits_1;
    loval = -hival;

    /* data stream */
    for (nin = 1; nin < n; nin++) {
      inval = in[nin];
      exc = 0;

      /* out of range ? - note and force exception code */
      if (check_exc && (inval <= loval || inval >= hival)) {
        exc = 1;
        inval = loval;
      }

      /* copy the value to output (bitwise) */
      for (bitin = nbits_1; bitin >= 0; bitin--) {
        if (inval & setinbit[bitin])
          outval |= setoutbit[bitout];
        bitout--;
        if (bitout < 0) {
          out[nout++] = outval;
          bitout = 7;
          outval = 0;
        }
      }

      /* exception alert - write the full value */
      if (exc) {
        inval = in[nin];
        for (bitin = nexcbits_1; bitin >= 0; bitin--) {
          if (inval & setinbit[bitin])
            outval |= setoutbit[bitout];
          bitout--;
          if (bitout < 0) {
            out[nout++] = outval;
            bitout = 7;
            outval = 0;
          }
        }
      }
    }

    /* don't forget to write the last bits */
    if (bitout != 7) {
    /*  printf("bitout: %d\n", bitout);*/
      out[nout++] = outval;
    }
  }

  else if (method == RAW3_COPY) {
    nout = 1;
    for (nin = 0; nin < n; nin++) {
      inval=in[nin];
      out[nout++] = (inval >> 8);
      out[nout++] = inval;
    }
  }

  else if (method == RAW3_COPY_32) {
    nout = 1;
    for (nin = 0; nin < n; nin++) {
      out[nout++] = (in[nin] >> 24);
      out[nout++] = (in[nin] >> 16);
      out[nout++] = (in[nin] >> 8);
      out[nout++] = in[nin];
    }
  }

  /* for test only
  printf("\nnout: %d\n", nout);
  */

  return nout;
}

int dehuffman16(unsigned char *in, int n, int *method, sraw_t *out)
{
  int  nin = 0, nout = 0;
  int  nbit, nbit_1, nexcbit, nexcbit_1, check_exc;
  int  bitin;
  sraw_t excval;
  /*int i,j; */

  int hibytein;
  unsigned int iwork;
  sraw_t swork;

  static sraw_t negmask[33] = {
    0xffffffff, 0xfffffffe, 0xfffffffc, 0xfffffff8,
    0xfffffff0, 0xffffffe0, 0xffffffc0, 0xffffff80,
    0xffffff00, 0xfffffe00, 0xfffffc00, 0xfffff800,
    0xfffff000, 0xffffe000, 0xffffc000, 0xffff8000,
    0xffff0000, 0xfffe0000, 0xfffc0000, 0xfff80000,
    0xfff00000, 0xffe00000, 0xffc00000, 0xff800000,
    0xff000000, 0xfe000000, 0xfc000000, 0xf8000000,
    0xf0000000, 0xe0000000, 0xc0000000, 0x80000000,
    0x00000000
  };

  static sraw_t posmask[33] = {
    0x00000000, 0x00000001, 0x00000003, 0x00000007,
    0x0000000f, 0x0000001f, 0x0000003f, 0x0000007f,
    0x000000ff, 0x000001ff, 0x000003ff, 0x000007ff,
    0x00000fff, 0x00001fff, 0x00003fff, 0x00007fff,
    0x0000ffff, 0x0001ffff, 0x0003ffff, 0x0007ffff,
    0x000fffff, 0x001fffff, 0x003fffff, 0x007fffff,
    0x00ffffff, 0x01ffffff, 0x03ffffff, 0x07ffffff,
    0x0fffffff, 0x1fffffff, 0x3fffffff, 0x7fffffff,
    0xffffffff
  };

  static sraw_t setbit[16] = {
    0x0001, 0x0002, 0x0004, 0x0008,
    0x0010, 0x0020, 0x0040, 0x0080,
    0x0100, 0x0200, 0x0400, 0x0800,
    0x1000, 0x2000, 0x4000, 0x8000
  };

  /* for test only
  printf("dehuffman16: ");
  */

  *method = (in[0] >> 4) & 0x0f;

  if (*method != RAW3_COPY) {
    nbit = in[0] & 0x0f;
    /* using 4-bit coding nbit=16 is coded as zero */
    if (nbit == 0) {
      nbit = 16;
      if( CheckVerbose ) {
         fprintf(stderr,"\nlibeep: critical compression method encountered "
                     "(method %d, 16 bit)\n", *method);
       }
      raw3_set_ERR_FLAG_16(1); 	 		
    } 		
    nbit_1 = nbit - 1;
    nexcbit = (in[1] >> 4) & 0x0f;
    /* using 4-bit coding nexcbit=16 is coded as zero */
    if (nexcbit == 0) nexcbit = 16;
    /* for test only
    printf("\nnbit: %d, nexcbit: %d, method: %d\n",nbit,nexcbit,*method);
    */
    nexcbit_1 = nexcbit - 1;
    excval = -(1 << (nbit_1));
    check_exc = (nbit != nexcbit);

    out[0] =   ((sraw_t) in[1] & 0x0f) << 12
             | (sraw_t) in[2] << 4
             | (((sraw_t) in[3] >> 4) & 0x0f);
    if (out[0] & 0x8000)  out[0] |= 0xffff0000;

    bitin = 28;
    nout = 1;

    /* we need to read 2 or 3 bytes from input to get all input value bits */
    if (nbit < 9) {
      while (nout < n) {
	/* index of first byte in inbytes containing bits of current value */
	hibytein = bitin >> 3;

	/* max 2 inbytes are needed here for all the bits */
	iwork = (in[hibytein] << 8) + in[hibytein + 1];
	swork = iwork >> (((hibytein + 2) << 3) - bitin - nbit);

	/* handle the sign in the work buffer */
	if (swork & setbit[nbit_1]) {
          swork |= negmask[nbit];
	}
	else {
          swork &= posmask[nbit];
	}

	bitin += nbit;

	/* exception ? - forget what we got and read again */
	if (swork == excval && check_exc) {
	  /* index of first byte containing bits of current exc. value */
	  hibytein = bitin >> 3;

	  /* max 3 inbytes are needed for all the exc. bits */
	  iwork = (in[hibytein] << 16) + (in[hibytein + 1] << 8) + in[hibytein + 2];
          swork = iwork >> (((hibytein + 3) << 3) - bitin - nexcbit);
	  /* handle the sign in the work buffer */
	  if (swork & setbit[nexcbit_1]) {
            swork |= negmask[nexcbit];
	  }
	  else {
            swork &= posmask[nexcbit];
	  }
	  bitin += nexcbit;
	}

	out[nout++] = swork;
      }
    }

    else {
      while (nout < n) {
	/* index of first byte in inbytes containing bits of current value */
	hibytein = bitin >> 3;

	/* max 3 inbytes are needed for all the bits */
	iwork = (in[hibytein] << 16) + (in[hibytein + 1] << 8) + in[hibytein + 2];
	swork = iwork >> (((hibytein + 3) << 3) - bitin - nbit);

	/* handle the sign in the work buffer */
	if (swork & setbit[nbit_1]) {
          swork |= negmask[nbit];
	}
	else {
          swork &= posmask[nbit];
	}

	bitin += nbit;

	/* exception ? - forget what we got and read again */
	if (swork == excval && nbit != nexcbit) {
	  /* index of first byte containing bits of current exc. value */
	  hibytein = bitin >> 3;

	  /* max 3 inbytes are needed for all the bits */
	  iwork = (in[hibytein] << 16) + (in[hibytein + 1] << 8) + in[hibytein + 2];
          swork = iwork >> (((hibytein + 3) << 3) - bitin - nexcbit);
	  /* handle the sign in the work buffer */
	  if (swork & setbit[nexcbit_1]) {
            swork |= negmask[nexcbit];
	  }
	  else {
            swork &= posmask[nexcbit];
	  }
	  bitin += nexcbit;
	}

	out[nout++] = swork;
      }
    }
    nin = bitin >> 3;
    if (bitin & 0x07) nin++;
  }
  /* RAW3_COPY mode */
  else {
    if( CheckVerbose ) {	
         fprintf(stderr,"\nlibeep: critical compression method encountered "
                   "(method 0 RAW3_COPY)\n");	
     }	
    raw3_set_ERR_FLAG_0(0);
    nin = 1;
    for (nout = 0; nout < n; nout++) {
      out[nout] = (((sraw_t) in[nin]) << 8) | in[nin + 1];
      /* read s16 instead of u16 */
      if(out[nout] > 32767) out[nout] -= 65536;
      nin += 2;
    }
  }

  /*
  int i;
  printf("%d | %d | %d |%d | %d\n", *method, nbit, nexcbit, nin, out[0]);
  for(i = 0; i < n; i++) {
    fprintf(stderr, "%d ", out[i]);
  }
  fprintf(stderr, "\n");
  */

  return nin;
}

int dehuffman32(unsigned char *in, int n, int *method, sraw_t *out)
{
  int nin = 0, nout = 0;
  int  nbit, nbit_1, nexcbit, nexcbit_1, check_exc;
  int  bitin, bitout;
  char inval; sraw_t outval, excval;
  /* int i; */

  static unsigned char setinbit[8] = {
    0x01, 0x02, 0x04, 0x08,
    0x10, 0x20, 0x40, 0x80
  };

  static unsigned int setoutbit[32] = {
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000
  };

  static unsigned int negmask[33] = {
    0xffffffff, 0xfffffffe, 0xfffffffc, 0xfffffff8,
    0xfffffff0, 0xffffffe0, 0xffffffc0, 0xffffff80,
    0xffffff00, 0xfffffe00, 0xfffffc00, 0xfffff800,
    0xfffff000, 0xffffe000, 0xffffc000, 0xffff8000,
    0xffff0000, 0xfffe0000, 0xfffc0000, 0xfff80000,
    0xfff00000, 0xffe00000, 0xffc00000, 0xff800000,
    0xff000000, 0xfe000000, 0xfc000000, 0xf8000000,
    0xf0000000, 0xe0000000, 0xc0000000, 0x80000000,
    0x00000000
  };

  *method = (in[0] >> 4) & 0x0f;

  if (*method != RAW3_COPY_32) {
    nbit = ((in[0] << 2) & 0x3c) | ((in[1] >> 6) & 0x03);
    nbit_1 = nbit - 1;
    nexcbit = in[1] & 0x3f;
    /* if (nexcbit == 0) nexcbit = 64; */
    nexcbit_1 = nexcbit - 1;

    out[0] =   ((sraw_t) in[2] << 24)
             | ((sraw_t) in[3] << 16)
             | ((sraw_t) in[4] <<  8)
             | ((sraw_t) in[5]);

    nin = 6;
    bitin = 7;
    inval = in[nin];
    nout = 1;
    bitout = nbit_1;
    outval = 0;
    excval = -(1 << (nbit_1));
    check_exc = (nbit != nexcbit);

    while (nout < n) {

      /* transfer bitwise from in to out */
      if (inval & setinbit[bitin]) {
        outval |= setoutbit[bitout];
        /* handle the sign in out according to sign bit in in */
        if (bitout == nbit_1) {
          outval |= negmask[nbit];
        }
      }

      bitin--;
      if (bitin < 0) {
        nin++;
        bitin = 7;
        inval = in[nin];
      }

      bitout--;

      /* is a residual complete ? */
      if (bitout < 0) {

        /* its an exception ? - forget this and read the large value which follows */
        if (outval == excval && check_exc) {
          outval = 0;
          for (bitout = nexcbit_1; bitout >= 0; bitout--) {
            if (inval & setinbit[bitin]) {
              outval |= setoutbit[bitout];
              if (bitout == nexcbit_1) {
                outval |= negmask[nexcbit];
              }
            }
            bitin--;
            if (bitin < 0) {
              nin++;
              bitin = 7;
              inval = in[nin];
            }
          }
        }

        /* now we really have the value, store it and set up for the next one */
        out[nout++] = outval;
        outval = 0;
        bitout = nbit - 1;
      }
    }
    /* count last incomplete input byte */
    if (bitin != 7) nin++;
  }

  /* RAW3_COPY_32 */
  else {
    nin = 1;
    for (nout = 0; nout < n; nout++) {
      out[nout] =   ((sraw_t) in[nin]   << 24)
                  | ((sraw_t) in[nin+1] << 16)
                  | ((sraw_t) in[nin+2] <<  8)
                  | ((sraw_t) in[nin+3]      );
      nin += 4;
    }
  }

  /*
  fprintf(stderr, "%d\t%d\t%ld\t%d\t%d\n", nbit, nexcbit, nin, *method, out[0]);
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%d ", out[i]);
  }
  fprintf(stderr, "\n");
  */

  return nin;
}

int dehuffman(unsigned char *in, int n, int *method, sraw_t *out)
{
  /* method bit 3 indicates 16 or 32 bit compression */
  if (in[0] & (unsigned char) 0x80) {
    return dehuffman32(in, n, method, out);
  }
  else {
    return dehuffman16(in, n, method, out);
  }
}

/* ---------------------------------------------------------------------
  take native raw data vectors (neighbors)
  calc the residuals using the supported methods
  find the best among them and compress
  write the compressed output to buffer
*/
int compchan(raw3_t *raw3, sraw_t *last, sraw_t *cur, int n, char *out)
{
  int i, imin, mi, short_method = RAW3_COPY;
  int sample;
  int lmin, length;
  raw3res_t *rc;
  sraw_t *res, *restime=NULL;
  int *hst;

  /* don't care about short vectors in each loop - force copy instead */
  if (n < 8) {

    short_method = RAW3_COPY;

    for (i = 0; i < n; i++) {
      if (cur[i] >= 32768 || cur[i] < -32768) {
        short_method = RAW3_COPY_32;
        break;
      }
    }
    length = huffman(cur, n, short_method, 0, 0, (unsigned char *) out);
  }

  /* calc residuals and their bit distribution
     using different methods
     select the best one and compress
  */
  else {
    for (mi = 0; mi < RAW3_METHODC; mi++) {

      /* prepare access to residual data table for this method */
      rc = &raw3->rc[mi];
      res = rc->res;
      hst = rc->hst;
      memset(hst, 0, 33 * sizeof(int));

      switch (mi) {
	case 0:
          rc->method = RAW3_TIME;
          res[0] = cur[0];
          for (sample = 1; sample < n; sample++)
            hst[bitc(res[sample] = cur[sample] - cur[sample - 1])]++;

          rc->length = huffman_size(hst, n - 1, &rc->nbits, &rc->nexcbits);
          restime = res;
          break;

	case 1:
          rc->method = RAW3_TIME2;
          res[0] = cur[0];	
          hst[bitc(res[1] = restime[1])]++;
          for (sample = 2; sample < n; sample++)
            hst[bitc(res[sample] = restime[sample] - restime[sample - 1])]++;

          rc->length = huffman_size(hst, n - 1, &rc->nbits, &rc->nexcbits);
          break;

	case 2:
          rc->method = RAW3_CHAN;
          res[0] = cur[0];
          for (sample = 1; sample < n; sample++)
            hst[bitc(res[sample] = restime[sample] - last[sample] + last[sample - 1])]++;

          rc->length = huffman_size(hst, n - 1, &rc->nbits, &rc->nexcbits);
          break;
      }
    }

    /* find the best residual method */
    lmin = 1000000;
    for (mi = 0; mi < RAW3_METHODC; mi++) {
    /*
      printf("\nmethod: %d, nbits: %d, nexcbits: %d, length: %d\n",
        raw3->rc[mi].method, raw3->rc[mi].nbits, raw3->rc[mi].nexcbits,
        raw3->rc[mi].length);
    */
      if (raw3->rc[mi].length < lmin) {
        imin = mi;
        lmin = raw3->rc[mi].length;
      }
    }
    rc = &raw3->rc[imin];

    /* need 32 bit storage ? */
    if (rc->nexcbits >= 16 || rc->res[0] < -32768 || rc->res[0] >= 32768 )
    {
      rc->method |= 0x08;
      short_method = RAW3_COPY_32;
    }
    /* is the best compression better than store only ? */
    if (rc->nbits < 32 && lmin + 3 < n * 4) {
      length = huffman(rc->res, n, rc->method, rc->nbits, rc->nexcbits,
        	        (unsigned char *) out);
      short_method = -1;
     }
    else
      length = huffman(cur, n, short_method, 0, 0,
        	        (unsigned char *) out);

  }

 /*
  if(short_method < 0)
  printf("\nmethod: %d, nbits: %d, nexcbits: %d, length: %d\n",
      rc->method, rc->nbits, rc->nexcbits, length);
  else printf("\nmethod: %d\n", short_method);

 */
  return length;
}

int decompchan(raw3_t *raw3, sraw_t *last, sraw_t *cur, int n, char *in)
{
  int method, sample, sample_1, sample_2;
  int length;
  sraw_t *res = raw3->rc[0].res;

  /* restore the residuals */

  length = dehuffman((unsigned char *) in, n, &method, res);

  /* build the values using residuals and method */

  switch (method & 0x07) {
    case RAW3_TIME:
      cur[0] = res[0];
      sample_1 = 0;
      for (sample = 1; sample < n; sample++) {
        cur[sample] = cur[sample_1] + res[sample];
        sample_1++;
      }
      break;

    case RAW3_TIME2:
      cur[0] = res[0];
      cur[1] = cur[0] + res[1];
      sample_2 = 0;
      sample_1 = 1;
      for (sample = 2; sample < n; sample++) {
        cur[sample] =
          2 * cur[sample_1] - cur[sample_2] + res[sample];
        sample_1++;
        sample_2++;
      }
      break;

    case RAW3_CHAN:
      cur[0] = res[0];
      sample_1 = 0;
      for (sample = 1; sample < n; sample++) {
        cur[sample] = cur[sample_1] + last[sample] - last[sample_1] + res[sample];
        sample_1++;
      }
      break;

    case RAW3_COPY:
      memcpy(cur, res, n * sizeof(sraw_t));
      break;

    default:
#if !defined(WIN32) || defined(__CYGWIN__)
      fprintf(stderr, "raw3: unknown compression method!\n");
#endif
      break;
  }

  return length;
}

int compepoch_mux(raw3_t *raw3, sraw_t *in, int length, char *out)
{
  int chan;
  int sample;
  sraw_t *tmp, *chanbase, *cur, *last;
  int samplepos;
  int outsize = 0, outsizealt;

  cur = raw3->cur;
  last = raw3->last;
  memset(last, 0, length * sizeof(sraw_t));

  for (chan = 0; chan < raw3->chanc; chan++) {

    /* extract one vector from MUX buffer */
    chanbase = &in[raw3->chanv[chan]];
    samplepos = 0;
    for (sample = 0; sample < length; sample++) {
      cur[sample] = chanbase[samplepos];
      samplepos += raw3->chanc;
    }
    /* calculate compression and its statistics */
    outsizealt = outsize;
    outsize += compchan(raw3, last, cur, length, &out[outsize]);
    decompchan(raw3,last,cur,length,&out[outsizealt]);
    /* prepare for reading next channel */
    tmp = cur; cur = last; last = tmp;
  }

  return outsize;
}

int decompepoch_mux(raw3_t *raw3, char *in, int length, sraw_t *out)
{
  int chan;
  int sample;
  sraw_t *tmp, *chanbase, *last, *cur;
  int samplepos;
  int insize = 0;

  cur = raw3->cur;
  last = raw3->last;
  memset(last, 0, length * sizeof(sraw_t));

  for (chan = 0; chan < raw3->chanc; chan++) {

    /* uncompress */
    insize += decompchan(raw3, last, cur, length, &in[insize]);

    /* mangle into MUX buffer */
    chanbase = &out[raw3->chanv[chan]];
    samplepos = 0;
    for (sample = 0; sample < length; sample++) {
      chanbase[samplepos] = cur[sample];
      samplepos += raw3->chanc;
    }

    /* prepare for reading next channel */
    tmp = cur; cur = last; last = tmp;
  }

  return insize;
}

void compchanv_mux(sraw_t *buf, int length,
                   short chanc, short *chanv)
{
  int chan, i, j, jmax, sample;
  float **rvv, rmax;
  float sumi, sumii, sumj, sumjj, sumij, x, y;

  /* create a [chanc x chanc] square matrix */
  rvv = (float **) malloc(chanc * sizeof(float *));
  for (i = 0; i < chanc; i++)
    rvv[i] = (float *) malloc(chanc * sizeof(float));

  /* calc a full matrix of correlation coefficients */
  for (i = 0; i < chanc; i++) {
    for (j = 0; j <= i; j++) {
      if (i == j) {
        rvv[i][j] = 1.0;
      }
      else {

        /* we need sums, squaresums, cosums */
        sumi = sumii = sumj = sumjj = sumij = 0.0;
        for (sample = 0; sample < length; sample++) {
          sumi += x = buf[i + chanc * sample];
          sumii += x * x;
          sumj += y = buf[j + chanc * sample];
          sumjj += y * y;
          sumij += x * y;
        }
        sumi /= length; sumii /= length;
        sumj /= length; sumjj /= length;
        sumij /= length;

        /* calc the correlation coeff. from sums, avoid fp exceptions here */
        x = (sumii - sumi * sumi) * (sumjj - sumj * sumj);
        y = x > 0.0 ? sqrt(x) : 0.0;
        rvv[i][j] = y > 1e-6 ? (sumij - sumi * sumj) / y : 0.0;
        rvv[j][i] = rvv[i][j];
      }
    }
  }

  /* find a channel-similar channel-... path trough the matrix */
  chanv[0] = 0;
  for (chan = 1; chan < chanc; chan++) {
    /* mark used channels in matrix - set column to a unreachable min correlation*/
    for (i = 0; i < chanc; i++)
      rvv[i][chanv[chan - 1]] = -2.0;

    /* find next chan to use - most similar to last -> max. correlation in row*/
    rmax = -2.0;
    i = chanv[chan - 1];
    for (j = 0; j < chanc; j++) {
      if (rvv[i][j] > rmax) {
        rmax = rvv[i][j];
        jmax = j;
      }
    }

    /* register it */
    chanv[chan] = jmax;
  }

  for(i = 0; i < chanc; i++)
    free((char *) rvv[i]);
  free((char *) rvv);
}

raw3_t *raw3_init(int chanc, short *chanv, uint64_t length)
{
  int i;
  raw3_t *raw3 = (raw3_t *) malloc(sizeof(raw3_t));

  if (!raw3)
    return NULL;

  raw3->chanc = chanc;
  raw3->chanv = (short *) malloc(chanc * sizeof(sraw_t));

  for (i = 0; i < 3; i++)
    raw3->rc[i].res = (sraw_t *) malloc(length * sizeof(sraw_t));
  raw3->last = (sraw_t *) malloc(length * sizeof(sraw_t));
  raw3->cur = (sraw_t *) malloc(length * sizeof(sraw_t));

  if (!raw3->cur || !raw3->last || !raw3->chanv) {
    raw3_free(raw3);
    return NULL;
  }
  memcpy(raw3->chanv, chanv, chanc * sizeof(short));

  return raw3;
}

void    raw3_free(raw3_t *raw3)
{
  int i;

  if (raw3) {
    if (raw3->chanv) free(raw3->chanv);
    for (i = 0; i < 3; i++)
      if (raw3->rc[i].res) free(raw3->rc[i].res);
    if (raw3->last) free(raw3->last);
    if (raw3->cur) free(raw3->cur);
    free(raw3);
  }
}
