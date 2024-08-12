// system
#include <inttypes.h>
#include <stdlib.h>
// libeep
#include <eep/eepio.h>
#include <cnt/cnt.h>
///////////////////////////////////////////////////////////////////////////////
void
handle_file(const char *filename) {
  /*********************************
   * variables we need for reading *
   *********************************/
  int i,status;
  FILE  * _libeep_file;            // file access.
  eeg_t * _libeep_avr;
  float * _libeep_muxbuf;          // buffer for sample data
  fprintf(stderr, "handling %s...\n", filename);
  /**************
   * initialize *
   **************/
  _libeep_file=eepio_fopen(filename, "rb");
  if(_libeep_file == NULL) {
    fprintf(stderr, "could not open %s\n", filename);
    return;
  }
  _libeep_avr=eep_init_from_file(filename, _libeep_file, &status);
  if(_libeep_avr == NULL || status != CNTERR_NONE) {
    fprintf(stderr, "could not initialize %s\n", filename);
    return;
  }
  /*******************
   * print some info *
   *******************/
  printf("sampling rate........ %f\n", 1.0 / eep_get_period(_libeep_avr));
  printf("number of samples.... %" PRIu64 "\n", eep_get_samplec(_libeep_avr));
  printf("number of channels... %i\n", eep_get_chanc(_libeep_avr));
  printf("history.............. %s\n", eep_get_history(_libeep_avr));
  printf("pre-stim interval.... %f\n", eep_get_pre_stimulus_interval(_libeep_avr));
  printf("trials(total)........ %li\n", eep_get_total_trials(_libeep_avr));
  printf("trials(averaged)..... %li\n", eep_get_averaged_trials(_libeep_avr));
  printf("condition label...... %s\n", eep_get_conditionlabel(_libeep_avr));
  printf("condition color...... %s\n", eep_get_conditioncolor(_libeep_avr));
  /******************
   * print channels *
   ******************/
  for(i=0;i<eep_get_chanc(_libeep_avr);i++) {
    printf("  channel(%i): %s, %s, scaling: %f\n",
      i,
      eep_get_chan_label(_libeep_avr, i),
      eep_get_chan_unit(_libeep_avr, i),
      eep_get_chan_scale(_libeep_avr, i));
  }
  /****************
   * print sample *
   ****************/
  // as a demo, read one sample at our favourite position in the file: offset 13
  // first, allocate memory(to hold 1 sample)
  _libeep_muxbuf = (float*)(malloc(FLOAT_CNTBUF_SIZE(_libeep_avr, 1)));
  // seek to offset
  eep_seek(_libeep_avr, DATATYPE_AVERAGE, 13, 0);
  // read sample
  eep_read_float(_libeep_avr, DATATYPE_AVERAGE, _libeep_muxbuf, 1);
  printf("sample 13: ");
  for(i=0;i<eep_get_chanc(_libeep_avr);i++) {
    printf("%f ", _libeep_muxbuf[i] * eep_get_chan_scale(_libeep_avr, i));
  }
  printf("\n");
  /***********
   * cleanup *
   ***********/
  eep_free(_libeep_avr);
  free(_libeep_muxbuf);
  eepio_fclose(_libeep_file);
}
///////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv) {
  int i;
  for(i=1;i<argc;i++) {
    handle_file(argv[i]);
  }
  return 0;
}
