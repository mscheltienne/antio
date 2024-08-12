/* system */
#include <stdio.h>
/* libeep */
#include <cnt/seg.h>
/*****************************************************************************/
int
main(int argc, char ** argv) {
  int            i;
  libeep_seg_t * s;
  for(i=1;i<argc;++i) {
    fprintf(stderr, "--- file: %s ---\n", argv[i]);
    s=libeep_seg_read(argv[i]);
    // TODO
    libeep_seg_delete(s);
  }
  return 0;
}
