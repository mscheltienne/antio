// system
#include <stdio.h>
// libeep
#include <v4/eep.h>
///////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv) {
  printf("libeep version: %s\n", libeep_get_version());
  return 0;
}
