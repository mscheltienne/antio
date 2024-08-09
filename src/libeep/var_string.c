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
#include <string.h>
#include <eep/var_string.h>

/* Create a new instance of simple_string.
   The string will be empty */
var_string varstr_construct()
{
  var_string s = (var_string) malloc(sizeof(struct var_string_s));
  if (NULL != s)
  {
    s->s = (char*) malloc(256 * sizeof(char));
    if (NULL == s->s)
    {
      free(s);
      s = NULL;
    }
    else
    {
      s->num_alloc = 256;
      s->length = 0;
      s->s[0] = '\0';
    }
  }
  return s;
}

/* Delete a string instance */
void varstr_destruct(var_string s)
{
  if (NULL != s)
  {
    if (NULL != s->s)
      free(s->s);
    free(s);
  }
}

/* Append a string to the existing string */
int varstr_append(var_string s, const char* app)
{
  long needed = s->num_alloc;
  long added_len = strlen(app);

  while (s->length + added_len >= needed)
    needed*=2;

  if (needed > s->num_alloc)
  { /* We have to alloc more space */
    char* new_s = (char*)realloc(s->s, needed);
    if (NULL == new_s)
      return 0; /* Can not enlarge string; the original should
                still be OK though */
    s->s = new_s; /* We now have twice as much space */
    s->num_alloc = needed;
  }
  /* Now append the string */
  strcat(s->s, app);
  s->length += added_len;
  return 1;
}

/* Set the string to this value */

int varstr_set(var_string s, const char* newstr)
{
  long new_len = strlen(newstr);
  /* round to nearest power of 2 for optimal allocs */
  long new_alloc = s->num_alloc;
  while (new_alloc < new_len)
    new_alloc*=2;
  if (new_alloc > s->num_alloc)
  {
    char *new_s = (char*)realloc(s->s, new_alloc);
    if (NULL == new_s)
      return 0;
    s->s = new_s;
    s->num_alloc = new_alloc;
  }
  strcpy(s->s, newstr);
  s->length = new_len;
  return 1;
}

/* Get the 'char*' representation of this string */

const char* varstr_cstr(var_string s)
{
  return s ? s->s : NULL;
}

/* Returns the current length of the string */
long varstr_length(var_string s)
{
  return s ? s->length : -1;
}
