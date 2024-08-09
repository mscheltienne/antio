// system
#include <stdlib.h>
#include <stdio.h>
// libeep
extern "C" {
  #include <v4/eep.h>
}
#include <libeep-jni.h>
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL
Java_com_antneuro_libeep_init(JNIEnv *, jclass) {
  libeep_init();
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL
Java_com_antneuro_libeep_exit(JNIEnv *, jclass) {
  libeep_exit();
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jstring JNICALL
Java_com_antneuro_libeep_get_1version(JNIEnv *env, jclass) {
  return env->NewStringUTF(libeep_get_version());
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jint JNICALL Java_com_antneuro_libeep_create_1channel_1info(JNIEnv *env, jclass) {
  return libeep_create_channel_info();
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jint JNICALL Java_com_antneuro_libeep_add_1channel(JNIEnv *env, jclass, jint channel_info_handle, jstring label, jstring ref, jstring unit) {
  const char *native_label = env->GetStringUTFChars(label, 0);
  const char *native_ref = env->GetStringUTFChars(ref, 0);
  const char *native_unit = env->GetStringUTFChars(unit, 0);
  int rv = libeep_add_channel(channel_info_handle, native_label, native_ref, native_unit);
  env->ReleaseStringUTFChars(label, native_label);
  env->ReleaseStringUTFChars(ref, native_ref);
  env->ReleaseStringUTFChars(unit, native_unit);
  return rv;
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jint JNICALL Java_com_antneuro_libeep_create_1recording_1info(JNIEnv *env, jclass) {
  return libeep_create_recinfo();
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL Java_com_antneuro_libeep_set_1start_1time(JNIEnv *env, jclass, jint handle, jlong epoch) {
  libeep_set_start_time(handle, epoch);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL Java_com_antneuro_libeep_set_1date_1of_1birth(JNIEnv *, jclass, jint handle, jint year, jint month, jint day) {
  libeep_set_date_of_birth(handle, year, month, day);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL Java_com_antneuro_libeep_set_1patient_1name(JNIEnv *env, jclass, jint handle, jstring name) {
  const char *native_name = env->GetStringUTFChars(name, 0);
  libeep_set_patient_name(handle, native_name);
  env->ReleaseStringUTFChars(name, native_name);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL Java_com_antneuro_libeep_add_1recording_1info(JNIEnv *, jclass, jint handle, jint recording_info_handle) {
  libeep_add_recording_info(handle, recording_info_handle);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jint JNICALL Java_com_antneuro_libeep_write_1cnt(JNIEnv *env, jclass, jstring filename, jint rate, jint channel_info_handle, jint rf64) {
  const char *native_filename = env->GetStringUTFChars(filename, 0);
  jint rv=libeep_write_cnt(native_filename, rate, channel_info_handle, rf64);
  env->ReleaseStringUTFChars(filename, native_filename);
  return rv;
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jint JNICALL Java_com_antneuro_libeep_read_1cnt (JNIEnv *env, jclass, jstring filename) {
  const char *native_filename = env->GetStringUTFChars(filename, 0);
  jint rv=libeep_read(native_filename);
  env->ReleaseStringUTFChars(filename, native_filename);
  return rv;
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL Java_com_antneuro_libeep_close(JNIEnv *, jclass, jint handle) {
  return libeep_close(handle);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jint JNICALL Java_com_antneuro_libeep_get_1channel_1count(JNIEnv *env, jclass, jint handle) {
  return libeep_get_channel_count(handle);
  fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jstring JNICALL Java_com_antneuro_libeep_get_1channel_1label(JNIEnv *env, jclass, jint handle, jint channel_id) {
  return env->NewStringUTF(libeep_get_channel_label(handle, channel_id));
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jstring JNICALL Java_com_antneuro_libeep_get_1channel_1unit(JNIEnv *env, jclass, jint handle, jint channel_id) {
  return env->NewStringUTF(libeep_get_channel_unit(handle, channel_id));
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jstring JNICALL Java_com_antneuro_libeep_get_1channel_1reference(JNIEnv *env, jclass, jint handle, jint channel_id) {
  return env->NewStringUTF(libeep_get_channel_reference(handle, channel_id));
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jfloat JNICALL Java_com_antneuro_libeep_get_1sample_1frequency(JNIEnv *, jclass, jint handle) {
  return libeep_get_sample_frequency(handle);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jlong JNICALL Java_com_antneuro_libeep_get_1sample_1count(JNIEnv *, jclass, jint handle) {
  return libeep_get_sample_count(handle);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jfloatArray JNICALL Java_com_antneuro_libeep_get_1samples(JNIEnv *env, jclass, jint handle, jlong from, jlong to) {
  int n=libeep_get_channel_count(handle);
  jfloatArray result = env->NewFloatArray(n*(to-from));
  if (result == NULL) {
    return NULL;
  }
  int i;
  // fill a temp structure to use to populate the java int array
  float * buf=libeep_get_samples(handle, from, to);
  jfloat fill[n*(to-from)];
  for(i = 0;i <(n*(to-from));i++) {
    fill[i] = buf[i];
  }
  free(buf);
  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, n*(to-from), fill);
  return result;
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL Java_com_antneuro_libeep_add_1samples(JNIEnv *env, jclass, jint handle, jfloatArray data, jint n) {
  jfloat* native_data = env->GetFloatArrayElements(data, 0);

  libeep_add_samples(handle, native_data, n);

  env->ReleaseFloatArrayElements(data, native_data, 0);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jlong JNICALL Java_com_antneuro_libeep_get_1zero_1offset(JNIEnv *, jclass, jint handle) {
  return libeep_get_zero_offset(handle);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jstring JNICALL Java_com_antneuro_libeep_get_1condition_1label(JNIEnv *env, jclass, jint handle) {
  return env->NewStringUTF(libeep_get_condition_label(handle));
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jstring JNICALL Java_com_antneuro_libeep_get_1condition_1color(JNIEnv *env, jclass, jint handle) {
  return env->NewStringUTF(libeep_get_condition_color(handle));
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jlong JNICALL Java_com_antneuro_libeep_get_1trials_1total(JNIEnv *, jclass, jint handle) {
  return libeep_get_trials_total(handle);
}
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jlong JNICALL Java_com_antneuro_libeep_get_1trials_1averaged(JNIEnv *, jclass, jint handle) {
  return libeep_get_trials_averaged(handle);
}
