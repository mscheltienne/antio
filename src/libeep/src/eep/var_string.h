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

#ifndef EEP_VAR_STRING_H
#define EEP_VAR_STRING_H

struct var_string_s
{
	char* s;
	long num_alloc;
	long length;
};

typedef struct var_string_s* var_string;

/* Create a new instance of simple_string.
   The string will be empty */
var_string varstr_construct();

/* Delete a string instance */
void varstr_destruct(var_string);

/* Append a string to the existing string */
int varstr_append(var_string, const char*);

/* Set the string to this value */

int varstr_set(var_string, const char*);

/* Get the 'char*' representation of this string.
   NEVER free() the result! (use varstr_destruct instead) */
const char* varstr_cstr(var_string);

/* Returns the current length of the string */
long varstr_length(var_string);

#endif /* #ifndef EEP_VAR_STRING_H */
