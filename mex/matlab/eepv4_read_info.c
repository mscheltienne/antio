/********************************************************************************
 *                                                                              *
 * this file is part of:                                                        *
 * libeep, the project for reading and writing avr/cnt eeg and related files    *
 *                                                                              *
 ********************************************************************************
 *                                                                              *
 * LICENSE:Copyright (c) 2003-2016,                                             *
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

#include "mex.h"        /* Matlab specific  */
#include "matrix.h"     /* Matlab specific  */

#include <v4/eep.h>     /* MPI-ANT specific */

void
mexFunction (int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {
  char         filename[256];
  cntfile_t    libeep_handle;
  const int    dimensions_1_1[] = {1, 1};
  const int    field_names_count = 7;
  const char * field_names[] = { "version", "channel_count", "channels", "sample_count", "sample_rate", "trigger_count", "triggers" };
  const int    channel_field_names_count = 2;
  const char * channel_field_names[] = { "label", "unit" };
  const int    trigger_field_names_count = 10;
  const char * trigger_field_names[] = { "offset_in_file", "seconds_in_file", "label", "duration", "type", "code", "condition", "description", "videofilename", "impedances" };
  int          c;
  int          t;
  const char * trigger_label;
  uint64_t     trigger_offset;
  double       sample_rate;
  struct libeep_trigger_extension trigger_extension;
  mxArray    * mx_channel_count;
  mxArray    * mx_channels;
  mxArray    * mx_sample_count;
  mxArray    * mx_sample_rate;
  mxArray    * mx_trigger_count;
  mxArray    * mx_triggers;

  if(nrhs!=1) {
    mexErrMsgTxt("Invalid number of input arguments");
  }

  // get parameters
  mxGetString(prhs[0], filename, 256);

  // init library
  libeep_init();

  // open
  libeep_handle = libeep_read_with_external_triggers(filename);
  if(libeep_handle == -1) {
    mexErrMsgTxt("Could not open file");
  }
  sample_rate = libeep_get_sample_frequency(libeep_handle);

  // create return structure
  plhs[0] = mxCreateStructArray(2, dimensions_1_1, field_names_count, field_names);

  // set version
  mxSetField(plhs[0], 0, field_names[0], mxCreateString(libeep_get_version()));

  // channel_count
  mx_channel_count = mxCreateDoubleMatrix(1,1,mxREAL);
  *mxGetPr(mx_channel_count) = (double)libeep_get_channel_count(libeep_handle);
  mxSetField(plhs[0], 0, field_names[1], mx_channel_count);
  // channels
  mx_channels = mxCreateStructMatrix(1, libeep_get_channel_count(libeep_handle), channel_field_names_count, channel_field_names);
  for(c=0;c<libeep_get_channel_count(libeep_handle);++c) {
      // copy
      mxSetField(mx_channels, c, "label", mxCreateString(libeep_get_channel_label(libeep_handle, c)));
      mxSetField(mx_channels, c, "unit", mxCreateString(libeep_get_channel_unit(libeep_handle, c)));
  }
  mxSetField(plhs[0], 0, field_names[2], mx_channels);
  // sample_count
  mx_sample_count = mxCreateDoubleMatrix(1,1,mxREAL);
  *mxGetPr(mx_sample_count) = (double)libeep_get_sample_count(libeep_handle);
  mxSetField(plhs[0], 0, field_names[3], mx_sample_count);
  // sample_rate
  mx_sample_rate = mxCreateDoubleMatrix(1,1,mxREAL);
  *mxGetPr(mx_sample_rate) = sample_rate;
  mxSetField(plhs[0], 0, field_names[4], mx_sample_rate);
  // trigger_count
  mx_trigger_count = mxCreateDoubleMatrix(1,1,mxREAL);
  *mxGetPr(mx_trigger_count) = (double)libeep_get_trigger_count(libeep_handle);
  mxSetField(plhs[0], 0, field_names[5], mx_trigger_count);
  // triggers
  mx_triggers = mxCreateStructMatrix(1, libeep_get_trigger_count(libeep_handle), trigger_field_names_count, trigger_field_names);
  for(t=0;t<libeep_get_trigger_count(libeep_handle);++t) {
    trigger_label = libeep_get_trigger_with_extensions(libeep_handle, t, &trigger_offset, &trigger_extension);
    {
      // copy
      if(trigger_extension.condition) {
        mxSetField(mx_triggers, t, "condition", mxCreateString(trigger_extension.condition));
      }
      if(trigger_extension.description) {
        mxSetField(mx_triggers, t, "description", mxCreateString(trigger_extension.description));
      }
      if(trigger_extension.videofilename) {
        mxSetField(mx_triggers, t, "videofilename", mxCreateString(trigger_extension.videofilename));
      }
      if(trigger_extension.impedances) {
        mxSetField(mx_triggers, t, "impedances", mxCreateString(trigger_extension.impedances));
      }

      mxSetField(mx_triggers, t, "offset_in_file",  mxCreateDoubleScalar(trigger_offset));
      mxSetField(mx_triggers, t, "seconds_in_file", mxCreateDoubleScalar((double)(trigger_offset) / sample_rate));
      mxSetField(mx_triggers, t, "label",           mxCreateString(trigger_label));
      mxSetField(mx_triggers, t, "duration",        mxCreateDoubleScalar(trigger_extension.duration_in_samples));
      mxSetField(mx_triggers, t, "type",            mxCreateDoubleScalar(trigger_extension.type));
      mxSetField(mx_triggers, t, "code",            mxCreateDoubleScalar(trigger_extension.code));
    }
  }
  mxSetField(plhs[0], 0, field_names[6], mx_triggers);

  // close
  libeep_close(libeep_handle);

  // exit
  libeep_exit();
}
