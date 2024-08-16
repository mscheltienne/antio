#ifndef __libeep_v4_eep_h__
#define __libeep_v4_eep_h__

// system
#include <time.h>
#include <stdint.h>

/**
 * @file v4/eep.h
 * @brief v4 of the libeep header provides a simplified interface to the libeep
 * project. It exposes no details of the internal structure
 *****/

/*
Handle for executing operations on a CNT file.
A handle of this type can be obtained in one of two ways:
- by a call to libeep_read() in which case the CNT file will be opened for reading an no write operations shall be executed using this handle
- or by a call to libeep_write_cnt() in which case the CNT file will be opened for writing an no read operations shall be executed using this handle
The handle obtained in this way shall be disposed by a call to libeep_close().
*/
typedef int cntfile_t;

/*
Handle for adding recording information to a CNT file.
A handle of this type can be obtained by a call to libeep_create_recinfo().
After that the handle can be used to construct gradually recording information data
using functions like libeep_set_patient_name(). After the desired information is stored
the handle shall be passed to the function libeep_write_cnt() in order to store
the data into the CNT file to be written.
The handle obtained in this way will be automatically disposed when libeep_exit() is executed.
*/
typedef int recinfo_t;

/*
Handle for adding channel information to a CNT file.
A handle of this type can be obtained by a call to libeep_create_channel_info().
After that the handle can be used to construct a set of channels using the function
libeep_add_channel(). After the desired information is stored the handle shall be passed
to the function libeep_write_cnt().
The handle obtained in this way will be automatically disposed when libeep_exit() is executed.
*/
typedef int chaninfo_t;

/**
 * @brief init library
 */
void libeep_init();
/**
 * @brief exit library
 */
void libeep_exit();
/**
 * @brief get library version
 * @return version(do not free this string)
 */
const char * libeep_get_version();
/**
 * @brief open file for reading
 * @param filename the filename to the CNT or AVR to open
 * @return -1 on error, handle otherwise
 */
cntfile_t libeep_read(const char *filename);
/**
 * @brief open file for reading, and load triggers from external .evt or .trg file if exist, otherwise, use internal trgs from cnt.
 * @param filename the filename to the CNT or AVR to open
 * @return -1 on error, handle otherwise
 */
cntfile_t libeep_read_with_external_triggers(const char *filename);
/**
 * @brief open cnt file for writing
 * @param filename the filename to the CNT or AVR to open
 * @param rate the sampling rate(in Hz)
 * @param channel_info_handle handle obtained by a call to libeep_create_channel_info (and eventually populated by calls to libeep_add_channel). Can not be invalid
 * @param rf64 if not zero, create 64-bit riff variant
 * @return -1 on error, handle otherwise
 */
cntfile_t libeep_write_cnt(const char *filename, int rate, chaninfo_t channel_info_handle, int rf64);
/**
 * @brief close data file
 * @param handle handle obtained by a call to either libeep_read() or libeep_write_cnt()
 */
void libeep_close(cntfile_t handle);
/**
 * @brief get the number of channels
 * @param handle handle obtained by a call to libeep_read()
 * @return the channel count in the specified CNT file
 */
int libeep_get_channel_count(cntfile_t handle);
/**
 * @brief get a channel label
 * @param handle handle obtained by a call to libeep_read()
 * @param index channel index
 * @return a channel label(do not free this string)
 */
const char * libeep_get_channel_label(cntfile_t handle, int index);
/**
 * @brief get a channel unit
 * @param handle handle obtained by a call to libeep_read()
 * @return a channel unit(do not free this string)
 */
const char * libeep_get_channel_status(cntfile_t handle, int index);
/**
 * @brief get a channel status
 * @param handle handle obtained by a call to libeep_read()
 * @return a channel status(do not free this string)
 */
const char * libeep_get_channel_type(cntfile_t handle, int index);
/**
 * @brief get a channel type
 * @param handle handle obtained by a call to libeep_read()
 * @return a channel type(do not free this string)
 */
const char * libeep_get_channel_unit(cntfile_t handle, int index);
/**
 * @brief get a channel reference
 * @param handle handle obtained by a call to libeep_read()
 * @return a channel reference(do not free this string)
 */
const char * libeep_get_channel_reference(cntfile_t handle, int index);
/**
 * @brief get the channel scaling
 * @param handle handle obtained by a call to libeep_read()
 * @return scaling of the channel
 */
float libeep_get_channel_scale(cntfile_t handle, int index);
/**
 * @brief get the channel index
 * @param handle handle obtained by a call to libeep_read()
 * @return an index to the channel identified by this name
 */
int libeep_get_channel_index(cntfile_t handle, const char *label);
/**
 * @brief get the sample frequency
 * @param handle handle obtained by a call to libeep_read()
 * @return sample frequency in Hz
 */
int libeep_get_sample_frequency(cntfile_t handle);
/**
 * @brief get the number of samples
 * @param handle handle obtained by a call to libeep_read()
 * @return the number of samples belonging to this recording
 */
long libeep_get_sample_count(cntfile_t handle);
/**
 * @brief get data samples
 * @param handle handle obtained by a call to libeep_read()
 * @param from the first sample to be returned
 * @param to the end sample to be returned
 * @return dynamically allocated array of samples or NULL on failure(Result should be freed with a call to libeep_free_samples)
 */
float * libeep_get_samples(cntfile_t handle, long from, long to);
/**
* @brief deallocates the buffer returned by libeep_get_samples
* @param data pointer to float array obtained by a call to libeep_get_samples()
*/
void libeep_free_samples(float *data);
/**
 * @brief add data samples
 * @param handle handle obtained by a call to libeep_write_cnt()
 * @param data pointer to float array
 * @param n number of items in array
 */
void libeep_add_samples(cntfile_t handle, const float *data, int n);
/**
* @brief add data samples
* @param handle handle obtained by a call to libeep_write_cnt()
* @param data pointer to int array
* @param n number of items in array
*/
void libeep_add_raw_samples(cntfile_t handle, const int32_t *data, int n);
/**
* @brief get data samples
* @param handle handle obtained by a call to libeep_read()
* @param from the first sample to be returned
* @param to the end sample to be returned
* @return dynamically allocated array of samples or NULL on failure(Result should be freed with a call to libeep_free_raw_samples)
*/
int32_t * libeep_get_raw_samples(cntfile_t handle, long from, long to);
/**
* @brief deallocates the buffer returned by libeep_get_raw_samples
* @param data pointer to float array obtained by a call to libeep_get_raw_samples()
*/
void libeep_free_raw_samples(int32_t *data);
/**
* @brief returns a handle to a new recording info object which can be passed to libeep_write_cnt()
*/
recinfo_t libeep_create_recinfo();
/**
* @brief add recording info to file
* @param recinfo the handle to a recording info structure
*/
void libeep_add_recording_info(cntfile_t cnt_handle, recinfo_t recinfo_handle);
/**
* @brief retrieves the start time of the recording
* @param handle handle obtained by a call to libeep_read()
* @returns time_t(0) if the file does not contain recording info
*/
time_t libeep_get_start_time(cntfile_t handle);
/**
* @brief sets the start time of the recording
* @param handle handle obtained by a call to libeep_create_recinfo()
* @param start_date receives [StartDate]-part of start time stamp
* @param start_fraction receives [StartFraction]-part of start time stamp
*/
void libeep_get_start_date_and_fraction(recinfo_t handle, double* start_date, double* start_fraction);
/**
* @brief sets the start time of the recording
* @param handle handle obtained by a call to libeep_create_recinfo()
* @param start_time the desired start time stamp
*/
void libeep_set_start_time(recinfo_t handle, time_t start_time);
/**
* @brief sets the start time of the recording
* @param handle handle obtained by a call to libeep_create_recinfo()
* @param start_date [StartDate]-part of the desired start time stamp
* @param start_fraction [StartFraction]-part of the desired start time stamp
*/
void libeep_set_start_date_and_fraction(recinfo_t handle, double start_date, double start_fraction);
/**
* @brief retrieves information about the hospital the recording was made
* @param handle handle obtained by a call to libeep_read()
* @returns the name of the hospital. the returned pointer shall not be free'd
*/
const char *libeep_get_hospital(cntfile_t handle);
/**
* @brief sets information about the hospital the recording was made
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_hospital(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the name of the test this recording belongs to
* @param handle handle obtained by a call to libeep_read()
* @returns the name test. the returned pointer shall not be free'd
*/
const char *libeep_get_test_name(cntfile_t handle);
/**
* @brief sets information about the name of the test this recording belongs to
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_test_name(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the test serial number
* @param handle handle obtained by a call to libeep_read()
* @returns the serial number of the test. the returned pointer shall not be free'd
*/
const char *libeep_get_test_serial(cntfile_t handle);
/**
* @brief sets information about the test serial number
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_test_serial(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the physician responsible for the recording
* @param handle handle obtained by a call to libeep_read()
* @returns the name of the physician. the returned pointer shall not be free'd
*/
const char *libeep_get_physician(cntfile_t handle);
/**
* @brief sets information about the physician responsible for the recording
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_physician(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the technician responsible for the recording
* @param handle handle obtained by a call to libeep_read()
* @returns the name of the technician. the returned pointer shall not be free'd
*/
const char *libeep_get_technician(cntfile_t handle);
/**
* @brief sets information about the technician responsible for the recording
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_technician(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the acquisition hardware
* @param handle handle obtained by a call to libeep_read()
* @returns the name of the hardware. the returned pointer shall not be free'd
*/
const char *libeep_get_machine_make(cntfile_t handle);
/**
* @brief sets information about the acquisition hardware
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_machine_make(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the model of the acquisition hardware
* @param handle handle obtained by a call to libeep_read()
* @returns the model. the returned pointer shall not be free'd
*/
const char *libeep_get_machine_model(cntfile_t handle);
/**
* @brief sets information about the model of the acquisition hardware
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_machine_model(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the serial number of the acquisition hardware
* @param handle handle obtained by a call to libeep_read()
* @returns the serial number. the returned pointer shall not be free'd
*/
const char *libeep_get_machine_serial_number(cntfile_t handle);
/**
* @brief sets information about the serial number of the acquisition hardware
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_machine_serial_number(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the name of the patient
* @param handle handle obtained by a call to libeep_read()
* @returns the name of the patient. the returned pointer shall not be free'd
*/
const char *libeep_get_patient_name(cntfile_t handle);
/**
* @brief sets information about the name of the patient
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_patient_name(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the identification number for this patient
* @param handle handle obtained by a call to libeep_read()
* @returns the patient id. the returned pointer shall not be free'd
*/
const char *libeep_get_patient_id(cntfile_t handle);
/**
* @brief sets information about the identification number for this patient
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_patient_id(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the address of the patient
* @param handle handle obtained by a call to libeep_read()
* @returns the address of the patient. the returned pointer shall not be free'd
*/
const char *libeep_get_patient_address(cntfile_t handle);
/**
* @brief sets information about the address of the patient
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_patient_address(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the phone number of the patient
* @param handle handle obtained by a call to libeep_read()
* @returns the patient's phone number. the returned pointer shall not be free'd
*/
const char *libeep_get_patient_phone(cntfile_t handle);
/**
* @brief sets information about the phone number of the patient
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_patient_phone(recinfo_t handle, const char *value);
/**
* @brief retrieves information about any comments about this recording
* @param handle handle obtained by a call to libeep_read()
* @returns the comments. the returned pointer shall not be free'd
*/
const char *libeep_get_comment(cntfile_t handle);
/**
* @brief sets information about any comments about this recording
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_comment(recinfo_t handle, const char *value);
/**
* @brief retrieves information about the sex of the patient
* @param handle handle obtained by a call to libeep_read()
* @returns the sex of the patient. Possible values are 'M' for male and 'F' for female
*/
char libeep_get_patient_sex(cntfile_t handle);
/**
* @brief sets information about the sex of the patient
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_patient_sex(recinfo_t handle, char value);
/**
* @brief retrieves information about the handedness of the patient
* @param handle handle obtained by a call to libeep_read()
* @returns the handedness of the patient. Possible values are 'L' for left, 'R' for right and 'M' for mixed
*/
char libeep_get_patient_handedness(cntfile_t handle);
/**
* @brief sets information about the handedness of the patient
* @param handle handle obtained by a call to libeep_create_recinfo()
*/
void libeep_set_patient_handedness(recinfo_t handle, char value);
/**
* @brief retrieves information about the patients date of birth
* @param year gregorian year
* @param month 1-12
* @param day 1-31
*/
void libeep_get_date_of_birth(cntfile_t handle, int * year, int * month, int *day);
/**
* @brief sets information about the patients date of birth
* @param year gregorian year
* @param month 1-12
* @param day 1-31
*/
void libeep_set_date_of_birth(recinfo_t handle, int year, int month, int day);
/**
* @brief inserts a trigger into the file
* @param handle handle obtained by a call to libeep_write_cnt()
* @param sample trigger insertion position
* @param code trigger label
*/
int libeep_add_trigger(cntfile_t handle, uint64_t sample, const char *code);
/**
* @brief returns the count of all triggers
* @param handle handle obtained by a call to libeep_read()
*/
int libeep_get_trigger_count(cntfile_t handle);
/**
* @brief returns the label of the trigger at certain position in the trigger table
* @param handle handle obtained by a call to libeep_read()
* @param handle trigger index in the trigger table
* @param sample the sample at which the trigger is positioned
*/
const char *libeep_get_trigger(cntfile_t handle, int idx, uint64_t *sample);
/**
*
*/
struct libeep_trigger_extension {
  int32_t      type;
  int32_t      code;
  uint64_t     duration_in_samples;
  const char * condition;
  const char * description;
  const char * videofilename;
  const char * impedances;
};
/**
* @brief returns the label of the trigger at certain position in the trigger table
* @param handle handle obtained by a call to libeep_read()
* @param handle trigger index in the trigger table
* @param sample the sample at which the trigger is positioned
* @param duration duration in samples of the trigger
*/
const char *libeep_get_trigger_with_extensions(cntfile_t handle, int idx, uint64_t *sample, struct libeep_trigger_extension * te);
/**
 * @brief get zero offset(averages only)
 * @param handle handle obtained by a call to libeep_read()
 * @return offset of sample where event occurred
 */
long libeep_get_zero_offset(cntfile_t handle);
/**
 * @brief get condition label(averages only)
 * @param handle handle obtained by a call to libeep_read()
 * @return condition label(do not free this string)
 */
const char * libeep_get_condition_label(cntfile_t handle);
/**
 * @brief get condition color(averages only)
 * @param handle handle obtained by a call to libeep_read()
 * @return condition color(do not free this string)
 */
const char * libeep_get_condition_color(cntfile_t handle);
/**
 * @brief get total number of trials(averages only)
 * @param handle handle obtained by a call to libeep_read()
 * @return total number of trials
 */
long libeep_get_trials_total(cntfile_t handle);
/**
 * @brief get averaged number of trials(averages only)
 * @param handle handle obtained by a call to libeep_read()
 * @return averaged number of trials
 */
long libeep_get_trials_averaged(cntfile_t handle);
/**
* @brief returns a handle to a new channel info object which can be passed to libeep_add_channel() and libeep_write_cnt()
*/
chaninfo_t libeep_create_channel_info();
/**
* @brief close channel info handle
*/
void libeep_close_channel_info(chaninfo_t);
/**
* @brief adds information about a channel
* @param handle handle obtained by a call to libeep_create_channel_info()
* @param label the label for this channel. cannot be NULL
* @param ref_label the reference label for this channel. can be NULL. the label "ref" is used if no value is provided
* @param unit the data unit for this channel. can be NULL. the unit "uV" is used if no value is provided
* @return the number of channels if succesfull; -1 on error
*/
int libeep_add_channel(chaninfo_t handle, const char *label, const char *ref_label, const char *unit);

#endif
