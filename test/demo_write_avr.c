// system
#include <stdlib.h>
#include <math.h>
// libeep
#include <eep/eepio.h>
#include <cnt/cnt.h>
///////////////////////////////////////////////////////////////////////////////
#ifndef M_PI
#define M_PI 3.14159
#endif
///////////////////////////////////////////////////////////////////////////////
void
handle_file(const char *filename) {
  /*********************************
   * variables we need for reading *
   *********************************/
  int i,n;
  FILE      * _libeep_file;        // file pointer
  eeg_t     * _libeep_avr;         // avr structure
  float     * _libeep_muxbuf;      // used for samples
  eegchan_t * channel_structure;
  fprintf(stderr, "handling %s...\n", filename);
  /******************************************************
   * we need a channel structure before we can continue *
   ******************************************************/
  channel_structure=eep_chan_init(10);   // four channels to start with
  eep_chan_set(channel_structure, 0, "chan0", 1, 1, "uV");
  eep_chan_set(channel_structure, 1, "chan1", 1, 1, "uV");
  eep_chan_set(channel_structure, 2, "chan2", 1, 1, "uV");
  eep_chan_set(channel_structure, 3, "chan3", 1, 1, "uV");
  eep_chan_set(channel_structure, 4, "chan4", 1, 1, "uV");
  eep_chan_set(channel_structure, 5, "chan5", 1, 1, "uV");
  eep_chan_set(channel_structure, 6, "chan6", 1, 1, "uV");
  eep_chan_set(channel_structure, 7, "chan7", 1, 1, "uV");
  eep_chan_set(channel_structure, 8, "chan8", 1, 1, "uV");
  eep_chan_set(channel_structure, 9, "chan9", 1, 1, "uV");

  _libeep_avr=eep_init_from_values(1.0/250.0, 10, channel_structure);
  if(_libeep_avr==NULL) {
    fprintf(stderr, "could not initialize %s\n", filename);
    return;
  }

  eep_set_total_trials(_libeep_avr, 1);               // 1 trial in total
  eep_set_averaged_trials(_libeep_avr, 1);            // 1 averaged trial
  eep_set_conditionlabel(_libeep_avr, "cond");        // condition name
  eep_set_conditioncolor(_libeep_avr, "color:8");     // TODO, list color table
  eep_set_pre_stimulus_interval(_libeep_avr, 0.1);    // 0.1 seconds

  _libeep_file=eepio_fopen(filename, "wb");
  if(_libeep_avr==NULL) {
    fprintf(stderr, "could not open %s\n", filename);
    return;
  }
  if(eep_create_file(_libeep_avr, filename, _libeep_file, NULL, 0, "demo tool") != CNTERR_NONE) {
    fprintf(stderr, "could not create %s\n", filename);
    return;
  }
  if(eep_prepare_to_write(_libeep_avr, DATATYPE_AVERAGE, 250, NULL) != CNTERR_NONE) {
    fprintf(stderr, "error preparing %s\n", filename);
    return;
  }
  // allocate memory
  _libeep_muxbuf=(float*)(malloc(FLOAT_CNTBUF_SIZE(_libeep_avr, 1)));
  if(_libeep_muxbuf==NULL) {
    fprintf(stderr, "allocation error");
    return;
  }
  /*******************************************
   * write samples(sine values, lots of 'm). *
   *******************************************/
  for(i=0;i<250;i++) {
    for(n=0;n<10;n++) {
      _libeep_muxbuf[n]=10.0 * sin( (double)(2*n*i) * M_PI / 250);
    }
    if(eep_write_float(_libeep_avr, _libeep_muxbuf, 1) != CNTERR_NONE) {
      fprintf(stderr, "sample write error");
      return;
    }
  }
  // write stddev samples
  if(eep_prepare_to_write(_libeep_avr, DATATYPE_STDDEV, 250, NULL) != CNTERR_NONE) {
    fprintf(stderr, "error preparing %s\n", filename);
    return;
  }
  for(i=0;i<250;i++) {
    for(n=0;n<10;n++) {
      _libeep_muxbuf[n]=10.0 * sin( (double)(2*n*i) * M_PI / 250);
    }
    if(eep_write_float(_libeep_avr, _libeep_muxbuf, 1) != CNTERR_NONE) {
      fprintf(stderr, "sample write error");
      return;
    }
  }
  /****************
   * close nicely *
   ****************/
  eep_finish_file(_libeep_avr);
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
