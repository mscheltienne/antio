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
#include <cnt/cntutils.h>

#define RET_ON_CNTERROR(x) do { int i; if (CNTERR_NONE != (i = (x)) ) return (i); } while (0);

int eep_read_float_channel(eeg_t *cnt, eep_datatype_e type, const char* name,  float* buffer, slen_t start, int length)
{
  int index;
  int count;
  float* data = (float*) malloc(FLOAT_CNTBUF_SIZE(cnt, 1));
  if(!data)
    return CNTERR_MEM;

  index = eep_get_chan_index(cnt, name);
  if(index == -1)
    return CNTERR_DATA;

  RET_ON_CNTERROR(eep_seek(cnt, type, start, 0));
  
  for(count = 0; count < length; count++)
  {
    RET_ON_CNTERROR(eep_read_float(cnt, type, data, 1));
    buffer[count] = data[index];
  }
 
  free(data);

  return CNTERR_NONE;
}

int eep_save_data_matrix_channels(eeg_t *avr, eep_datatype_e type, float **v, int samplec)
{
  int i,j;
  float* buffer;
  int chanc = eep_get_chanc(avr);
  RET_ON_CNTERROR(eep_prepare_to_write(avr, type, samplec, 0));

  buffer = (float*) malloc(FLOAT_CNTBUF_SIZE(avr,1));

  for(i=0; i < samplec; i++)
  {
    for(j=0; j< chanc; j++)
      buffer[j] = v[j][i];

    eep_write_float(avr, buffer, 1);
  }
  
  free(buffer);
  return CNTERR_NONE;
}

int eep_read_sraw_channel(eeg_t *cnt, eep_datatype_e type, const char* name,  sraw_t* buffer, slen_t start, int length)
{
  int index;
  int count;
  sraw_t* data = (sraw_t*) malloc(CNTBUF_SIZE(cnt, 1));

  if(!data)
    return CNTERR_MEM;

  index = eep_get_chan_index(cnt, name);
  if(index == -1)
    return CNTERR_DATA;
 
  RET_ON_CNTERROR(eep_seek(cnt, type, start, 0));

  for(count = 0; count < length; count++)
  {
    RET_ON_CNTERROR(eep_read_sraw(cnt, type, data, 1));
    buffer[count] = data[index];
  }

  free(data);

  return CNTERR_NONE;
}

slen_t eep_get_sample0(eeg_t* avr)
{
  return (int)(-1 * eep_get_pre_stimulus_interval(avr) * eep_get_rate(avr) - 0.5);
}

void eep_set_sample0(eeg_t* avr, slen_t sample0)
{
  eep_set_pre_stimulus_interval(avr, (-1.0 * (double)sample0 ) / eep_get_rate(avr));
}


void eep_copy_standard_avr_settings(eeg_t* dst, eeg_t* source)
{
  eep_set_total_trials(dst, eep_get_total_trials(source));
  eep_set_averaged_trials(dst, eep_get_averaged_trials(source));
  eep_set_conditionlabel(dst, eep_get_conditionlabel(source));
  eep_set_conditioncolor(dst, eep_get_conditioncolor(source));
  eep_set_pre_stimulus_interval(dst, eep_get_pre_stimulus_interval(source));
}


