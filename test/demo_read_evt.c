/* system */
#include <stdio.h>
/* libeep */
#include <cnt/cnt.h>
#include <cnt/evt.h>
/*****************************************************************************/
int
main(int argc, char ** argv) {
  int i;
  for(i=1;i<argc;++i) {
    libeep_evt_event_t * e;
    fprintf(stderr, "--- file: %s ---\n", argv[i]);
    libeep_evt_t * evt = libeep_evt_read(argv[i]);

    if(evt != NULL) {
      /* print */
      libeep_evt_header_print(&evt->header);
      e=evt->evt_list_first;
      while(e != NULL) {
        struct tm     time;
               double stamp_start = 42451.578067129631;

        libeep_evt_event_print(e);

        eep_get_time_struct(e->time_stamp.date, e->time_stamp.fraction, &time);
        printf("  stamp string.. %s\n", asctime(&time));
        printf("  stamp delta... %10.10f\n", e->time_stamp.date - stamp_start);

        e = e->next_event;
      }
      libeep_evt_delete(evt);
    }
  }
  return 0;
}
