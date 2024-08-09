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

#include <string.h>

#include <eep/eepraw.h>
#include <eep/eepio.h>


#ifdef COMPILE_RCS
char RCS_eepraw_h[] = RCS_EEPRAW_H;
char RCS_eepraw_c[] = "$RCSfile: eepraw.c,v $ $Revision: 2415 $";
#endif

#define SWAP32(x) \
        ((unsigned int)((((unsigned int)(x) & 0x000000ff) << 24) | \
                        (((unsigned int)(x) & 0x0000ff00) <<  8) | \
                        (((unsigned int)(x) & 0x00ff0000) >>  8) | \
                        (((unsigned int)(x) & 0xff000000) >> 24)))

#define SWAP16(x) \
        ((unsigned short int)((((unsigned short int)(x) & 0x00ff) << 8) | \
                              (((unsigned short int)(x) & 0xff00) >> 8)))

int _read_64(FILE *f, char *v) {
#if EEP_BYTE_ORDER == EEP_LITTLE_ENDIAN
  return eepio_fread(v, 8, 1, f);
#else
  register char c;
  if (!eepio_fread(v, 8, 1, f)) return 0;
  c = v[0]; v[0] = v[7]; v[7] = c;
  c = v[1]; v[1] = v[6]; v[6] = c;
  c = v[2]; v[2] = v[5]; v[5] = c;
  c = v[3]; v[3] = v[4]; v[4] = c;
  return 1;
#endif
}

int _read_32(FILE *f, char *v) {
#if EEP_BYTE_ORDER == EEP_LITTLE_ENDIAN
  return eepio_fread(v, 4, 1, f);
#else
  register char c;
  if (!eepio_fread(v, 4, 1, f)) return 0;
  c = v[0]; v[0] = v[3]; v[3] = c;
  c = v[1]; v[1] = v[2]; v[2] = c;
  return 1;
#endif
}

int _read_16(FILE *f, char *v) {
#if EEP_BYTE_ORDER == EEP_LITTLE_ENDIAN
  return eepio_fread(v, 2, 1, f);
#else
  register char c;
  if (!eepio_fread(v, 2, 1, f)) return 0;
  c = v[0]; v[0] = v[1]; v[1] = c;
  return 1;
#endif
}

int _write_64(FILE *f, const char *v) {
#if EEP_BYTE_ORDER == EEP_LITTLE_ENDIAN
  const char *tmp=v;
#else
  char tmp[8];
  tmp[0]=v[7];
  tmp[1]=v[6];
  tmp[2]=v[5];
  tmp[3]=v[4];
  tmp[4]=v[3];
  tmp[5]=v[2];
  tmp[6]=v[1];
  tmp[7]=v[0];
#endif
  return eepio_fwrite(tmp, 8, 1, f);
}

int read_u64(FILE *f, uint64_t *v) {
  return _read_64(f, (char *)v);
}

int write_u64(FILE *f, uint64_t v) {
  return _write_64(f, (char *)&v);
}

int read_s32(FILE *f, int *v) {
  return _read_32(f, (char *)v);
}

int read_u32(FILE *f, unsigned int *v) {
  return _read_32(f, (char *)v);
}

void swrite_u64(char *s, uint64_t v)
{
  s[0] = (char) v;
  s[1] = (char) (v >> 8);
  s[2] = (char) (v >> 16);
  s[3] = (char) (v >> 24);
  s[4] = (char) (v >> 32);
  s[5] = (char) (v >> 40);
  s[6] = (char) (v >> 48);
  s[7] = (char) (v >> 56);
}

void swrite_s32(char *s, int v)
{
  s[0] = (char) v;
  s[1] = (char) (v >> 8);
  s[2] = (char) (v >> 16);
  s[3] = (char) (v >> 24);
}

int write_s32 (FILE *f, int v)
{
  char out[4];
  swrite_s32(out, v);
  return eepio_fwrite((char *) out, 4, 1, f);
}

int write_u32 (FILE *f, unsigned int v)
{
  char out[4];
  swrite_s32(out, v);
  return eepio_fwrite((char *) out, 4, 1, f);
}

int read_u16 (FILE *f, int *v)
{
  unsigned char in[2];
  if (!eepio_fread((char *) in, 2, 1, f)) return 0;
  *v = (unsigned int) in[0] + (in[1] << 8);
  return 1;
}

int read_s16  (FILE *f, int *v)
{
  unsigned char in[2];
  if (!eepio_fread((char *) in, 2, 1, f)) return 0;

  /* negative value ? */
  if (in[1] & 0x80) {
    *v = -1 & ~0xffff;   /* lower 16 bit 0, other 1 */
    *v |= (int) in[0] + (in[1] << 8);
  }
  else {
    *v = (int) in[0] + (in[1] << 8);
  }
  return 1;
}

void swrite_s16(char *s, int v)
{
  s[0] = (char) v;
  s[1] = (char) (v >> 8);
}

int write_u16(FILE *f, int v)
{
  char out[2];
  swrite_s16(out, v);
  return eepio_fwrite((char *) out, 2, 1, f);
}

int write_s16(FILE *f, int v)
{
  char out[2];
  swrite_s16(out, v);
  return eepio_fwrite((char *) out, 2, 1, f);
}

int read_f32 (FILE *f, float *v) {
  return _read_32(f, (char *)v);
}

int write_f32(FILE *f, float v)
{
  register char *tmp = (char *) &v;
#if EEP_FLOAT_ORDER == EEP_BIG_ENDIAN
  register char c;
  c = tmp[0]; tmp[0] = tmp[3]; tmp[3] = c;
  c = tmp[1]; tmp[1] = tmp[2]; tmp[2] = c;
#endif
  return eepio_fwrite(tmp, 4, 1, f);
}

int read_f64(FILE *f, double *v) {
  return _read_64(f, (char *)v);
}

int write_f64(FILE *f, double v)
{
  register char *tmp = (char *) &v;
#if EEP_FLOAT_ORDER == EEP_BIG_ENDIAN
  register char c;
  c = tmp[0]; tmp[0] = tmp[7]; tmp[7] = c;
  c = tmp[1]; tmp[1] = tmp[6]; tmp[6] = c;
  c = tmp[2]; tmp[2] = tmp[5]; tmp[5] = c;
  c = tmp[3]; tmp[3] = tmp[4]; tmp[4] = c;
#endif
  return eepio_fwrite(tmp, 8, 1, f);
}

void swrite_f32  (char *s, float v)
{
  register char *tmp = (char *) &v;
#if EEP_FLOAT_ORDER == EEP_BIG_ENDIAN
  s[0] = tmp[3];
  s[1] = tmp[2];
  s[2] = tmp[1];
  s[3] = tmp[0];
#else
  memcpy(s, tmp, sizeof(float));
#endif
}

void sread_f32(char *s, float *v) {
  register char *tmp = (char *) v;
#if EEP_FLOAT_ORDER == EEP_BIG_ENDIAN
  tmp[0] = s[3];
  tmp[1] = s[2];
  tmp[2] = s[1];
  tmp[3] = s[0];
#else
  memcpy(tmp, s, sizeof(float));
#endif
}

void swrite_f64(char *s, double v)
{
  register char *tmp = (char *) &v;
#if EEP_FLOAT_ORDER == EEP_BIG_ENDIAN
  s[0] = tmp[7];
  s[1] = tmp[6];
  s[2] = tmp[5];
  s[3] = tmp[4];
  s[4] = tmp[3];
  s[5] = tmp[2];
  s[6] = tmp[1];
  s[7] = tmp[0];
#else
  memcpy(s, tmp, sizeof(double));
#endif
}

int vread_s16(FILE *f, sraw_t *buf, int n)
{
  register int j, status;
  register unsigned char *tmp = (unsigned char *) buf;
  
  status = eepio_fread(tmp, 2, n, f);
  if (status != n)
    return status;
  
  for (j = n - 1; j >= 0; j--) {
    buf[j] = ((int) tmp[2*j + 1] << 8) | tmp[2*j];
    if (buf[j] & 0x8000)
      buf[j] |= 0xffff0000;
  }
  return n;
}
  
int vwrite_s16(FILE *f, sraw_t *buf, int n)
{

  register int j;
  int  nr;
  register unsigned char *tmp = (unsigned char *) buf;

  for (j = 0; j < n; j++) {
    tmp[2*j] = (unsigned char) (buf[j]);
    tmp[2*j+1] = (unsigned char) (buf[j] >> 8);
  }
  nr = eepio_fwrite(tmp, 2, n, f);

  /* reestablish original byte order */
  for (j = 0; j < n; j++) {                                                     
    tmp[2*j] = (unsigned char) (buf[j]);                                       
    tmp[2*j+1] = (unsigned char) (buf[j] >> 8);                                
  }  

  return nr;
}

int vread_f32(FILE *f, float *buf, int n)
{
  register char *tmp = (char *) buf;

#if EEP_FLOAT_ORDER == EEP_LITTLE_ENDIAN
  return eepio_fread(tmp, 4, n, f);
#else
  
  int status = eepio_fread(tmp, 4, n, f);
  register int j;
  register char *w,c;

  if (status == n) {
    for (j = 0; j < n; j++) {
      w = &tmp[j*4];
      c = w[0]; w[0] = w[3]; w[3] = c;
      c = w[1]; w[1] = w[2]; w[2] = c;
    }
  }
  return status;
#endif
}
  
int vwrite_f32(FILE *f, float *buf, int n)
{
  register char *tmp = (char *) buf;

#if EEP_FLOAT_ORDER == EEP_BIG_ENDIAN
  register int j;
  int nr;
  register char *w,c;

  for (j = 0; j < n; j++) {
    w = &tmp[j*4];
    c = w[0]; w[0] = w[3]; w[3] = c;
    c = w[1]; w[1] = w[2]; w[2] = c;
  }
  nr = eepio_fwrite(tmp, 4, n, f);

  /* reestablish original byte order */
  for (j = 0; j < n; j++) {
    w = &tmp[j*4];
    c = w[0]; w[0] = w[3]; w[3] = c;
    c = w[1]; w[1] = w[2]; w[2] = c;
  }

  return nr; 
#endif

  return eepio_fwrite(tmp, 4, n, f);
}

int vread_s32(FILE *f, sraw_t *buf, int n)
{
  int i;

  i = eepio_fread(buf, 4, n, f);
  if (i != n)
    return i;

#if EEP_BYTE_ORDER == EEP_LITTLE_ENDIAN
  return n;
#else
  for (i=0;i<n;++i) {
    buf[i] = SWAP32(buf[i]);
  }
  return n;
#endif
}

