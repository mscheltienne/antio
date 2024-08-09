/*
  ------------------------------------------------------------------------
  EEP - ERP Evaluation Package
  
  very basic stuff

  Max-Planck-Institute of Cognitive Neuroscience Leipzig
  R. Nowagk 1997

  (Id: eepmisc.c,v 1.3 1999/07/05 06:51:32 nowagk Exp nowag)

 
  fixed RCS log strings
  2000/08/08 - MG
  

  ------------------------------------------------------------------------
  $Id: eepmisc.c 1762 2008-03-14 13:15:30Z rsmies $
  ------------------------------------------------------------------------
*/

#include <string.h>                                                        
#include <stdlib.h>                                                        
#include <ctype.h>  

#include <eep/eepmisc.h>

#ifdef COMPILE_RCS
char RCS_eepmisc_h[] = RCS_EEPMISC_H;
char RCS_eepmisc_c[] = "$RCSfile: eepmisc.c,v $ $Revision: 1762 $";
#endif

#if defined(__EMX__)
int strcasecmp(const char *s1, const char *s2)
{
  int i, r;
  char *buf1 = malloc(strlen(s1) + 1);
  char *buf2 = malloc(strlen(s2) + 1);
  
  i = 0;
  do {
    buf1[i] = tolower(s1[i]);
    i++;
  } while(s1[i-1]);
  
  i = 0;
  do {
    buf2[i] = tolower(s2[i]);
    i++;
  } while(s2[i-1]);
  
  r = strcmp(buf1, buf2);
  
  free(buf1); free(buf2);
  return r;
}
#endif

int strend(char *s, char *m)
{
  int l1, l2;
  int x;
  
  l1 = strlen(s);
  l2 = strlen(m);
  if (l2 > l1)
    x = 1;
  else 
    x = strcmp(s + l1 - l2, m);
  
  if (x)
    return 0;
  else
    return 1;
}

int asciiwrite(FILE *f, char *line)
{
  /* return nonzero on success */
  return (fputs(line, f) != EOF);
}

char *asciiread(FILE *f, char *line)
{
  return fgets(line, EEPIOMAX, f);
}
