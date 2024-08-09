// system
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// libeep
#include <v4/eep.h>
///////////////////////////////////////////////////////////////////////////////
#define CHANNEL_COUNT 10
const int    sampling_rate = 512;
const int    duration = 60;
const char * channel_names[CHANNEL_COUNT] = { "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9" };
const char * channel_units[CHANNEL_COUNT] = { "uV", "uV", "uV", "uV", "uV", "uV", "uV", "uV", "uV", "uV" };
///////////////////////////////////////////////////////////////////////////////
void
handle_file(const char *filename) {
  int cnt_handle, i, c;
  chaninfo_t channel_info_handle;
  recinfo_t  recording_info_handle;

  // setup channel information
  channel_info_handle = libeep_create_channel_info();
  for (i = 0; i < CHANNEL_COUNT; ++i) {
	  libeep_add_channel(channel_info_handle, channel_names[i], "ref", channel_units[i]);
  }

  // setup recording info
  recording_info_handle = libeep_create_recinfo();
  libeep_set_start_time(recording_info_handle, time(NULL));
  libeep_set_patient_handedness(recording_info_handle, 'R');
  libeep_set_patient_sex(recording_info_handle, 'M');
  libeep_set_patient_name(recording_info_handle, "John Doe");
  libeep_set_hospital(recording_info_handle, "Hospital");
  libeep_set_date_of_birth(recording_info_handle, 1950, 6, 28);

  cnt_handle = libeep_write_cnt(filename, sampling_rate, channel_info_handle, 1);
  if(cnt_handle == -1) {
    fprintf(stderr, "error opening %s", filename);
  }

  float * samples = (float *)malloc(sizeof(float) * CHANNEL_COUNT);
  for(i=0;i<sampling_rate * duration;++i) {
    for(c=0;c<CHANNEL_COUNT;++c) {
      samples[c] = (float)(i);
    }
    libeep_add_samples(cnt_handle, samples, 1);
    // for testing
  }
  free(samples);

  // add recording info
  libeep_add_recording_info(cnt_handle, recording_info_handle);

  libeep_close(cnt_handle);
}
///////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv) {
  libeep_init();

  int i;
  for(i=1;i<argc;i++) {
    handle_file(argv[i]);
  }

  libeep_exit();

  return 0;
}
