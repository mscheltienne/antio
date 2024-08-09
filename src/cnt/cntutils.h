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
#ifndef CNTUTILS_H
#define CNTUTILS_H

#include <cnt/cnt.h>

/* 
 Fill the buffer with data from one channel
 @param cnt the cnt/avr file from which we want data.
 @param type the datatype to get from the cnt/avr file.
 @param name the name of the channel we want data from.
 @param buffer the buffer in which to put data. Must be malloced before calling this.
 @param start the first sample, from which we want data.
 @param length the amonut of samples we want.
 */
int eep_read_float_channel(eeg_t *cnt, eep_datatype_e type, const char* name,  float* buffer, slen_t start, int length);
int eep_read_sraw_channel(eeg_t *cnt, eep_datatype_e type, const char* name,  sraw_t* buffer, slen_t start, int length);

/* 
 Set and get the sample0. 
 Mostly for use in old avr methods. Once there was a sample0 in the avr structure. 
 Sample0 is a negative number which identifies the first sample in the pre_stimulus part.
*/
slen_t eep_get_sample0(eeg_t* avr);
void eep_set_sample0(eeg_t* avr, slen_t sample0);


/*
 This function is a replacement for avr_save.
 It will put the data of v into the avrfile.
 @param avr the avrfile.
 @param type the datatype which will be written. There should be no such datatype in the avrfile.
 @param v the data in a matrix of chanc * samplec.
 @param samplec the number of samples.
 */
int eep_save_data_matrix_channels(eeg_t *avr, eep_datatype_e type, float **v, int samplec);
  

/*
 * In many cases you want to use the settings of an old avr file for a new created one.
 * This will set the trials/conditionlabel/conditioncolor/prestimulus_interval.
 */
void eep_copy_standard_avr_settings(eeg_t* dst, eeg_t* source);
  
#endif /* CNTUTILS_H */
