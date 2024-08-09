cmake_minimum_required(VERSION 2.8)

set(LIBEEP_VERSION_MAJOR 3)
set(LIBEEP_VERSION_MINOR 3)
set(LIBEEP_VERSION_PATCH 179)
set(LIBEEP_VERSION "${LIBEEP_VERSION_MAJOR}.${LIBEEP_VERSION_MINOR}.${LIBEEP_VERSION_PATCH}")
set(prefix ${CMAKE_INSTALL_PREFIX})
set(PACKAGE libeep)

project(libeep-${LIBEEP_VERSION})

add_definitions(-DLIBEEP_VERSION_MAJOR=${LIBEEP_VERSION_MAJOR})
add_definitions(-DLIBEEP_VERSION_MINOR=${LIBEEP_VERSION_MINOR})
add_definitions(-DLIBEEP_VERSION_PATCH=${LIBEEP_VERSION_PATCH})
if(UNIX)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=switch")
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  configure_file(libeep.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libeep.pc)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libeep.pc DESTINATION lib/pkgconfig)
endif()
if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DNOMINMAX)
  add_definitions(-DSTRICT)
  add_definitions(-D_USE_MATH_DEFINES)
  add_definitions(-DVC_EXTRALEAN)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  set(Eep_def windows/Eep.def)
endif(WIN32)

set(Eep_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libavr/avr.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libavr/avrcfg.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/cnt.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/cntutils.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/evt.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/raw3.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/rej.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/riff64.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/riff.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/seg.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libcnt/trg.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libeep/eepio.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libeep/eepmem.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libeep/eepmisc.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libeep/eepraw.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libeep/val.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libeep/var_string.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/v4/eep.c
)
add_library(EepObjects OBJECT
  ${Eep_sources}
)
target_include_directories(EepObjects PRIVATE src)
target_include_directories(EepObjects PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen/include)

add_library(EepStatic STATIC
  $<TARGET_OBJECTS:EepObjects>
)
target_include_directories(EepStatic PUBLIC src)

add_library(Eep SHARED
  $<TARGET_OBJECTS:EepObjects>
  ${Eep_def}
)
target_include_directories(Eep PUBLIC src)

install(TARGETS Eep DESTINATION lib)

install(FILES src/v4/eep.h DESTINATION include/libeep-${LIBEEP_VERSION}/v4)

########
# java #
########
if(NOT EXISTS $ENV{JAVA_HOME})
  message("please set the environment variable JAVA_HOME to build the JNI package")
else()
  add_subdirectory(java)
endif()

##########
# matlab #
##########
if(IS_DIRECTORY $ENV{MATLAB})
  if(IS_DIRECTORY $ENV{MATLAB}/bin/win64)
    # windows 64
    set(MATLAB_BIN $ENV{MATLAB}/bin/win64)
    set(MATLAB_EXT mexw64)
    set(MATLAB_MEX_CFLAGS -DMEX_CXXFLAGS=-ansi -DWIN32)
  elseif(IS_DIRECTORY $ENV{MATLAB}/bin/win32)
    # windows 32
    set(MATLAB_BIN $ENV{MATLAB}/bin/win32)
    set(MATLAB_EXT mexw32)
    set(MATLAB_MEX_CFLAGS -DMEX_CXXFLAGS=-ansi -DWIN32)
  elseif(IS_DIRECTORY $ENV{MATLAB}/bin/glnx32)
    # linux 32
    set(MATLAB_BIN $ENV{MATLAB}/bin/glnx32)
    set(MATLAB_EXT mexa32)
  elseif(IS_DIRECTORY $ENV{MATLAB}/bin/glnxa64)
    # linux 64
    set(MATLAB_BIN $ENV{MATLAB}/bin/glnxa64)
    set(MATLAB_EXT mexa64)
    set(MATLAB_MEX_CFLAGS "CFLAGS=-std=c99 -fPIC -D_GNU_SOURCE -O3 -v")
  else()
    # fallback, mac 64
    set(MATLAB_BIN $ENV{MATLAB}/bin)
    set(MATLAB_EXT mexmaci64)
  endif()

  message("Eep................. ${Eep}")
  message("MATLAB_BIN.......... ${MATLAB_BIN}")
  message("MATLAB_EXT.......... ${MATLAB_EXT}")

  foreach(matlab_target read_eep_avr read_eep_cnt read_eep_trg write_eep_avr write_eep_cnt eepv4_version eepv4_read_info eepv4_read)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mex/matlab/${matlab_target}.c)
      add_custom_command(OUTPUT ${matlab_target}.${MATLAB_EXT}
                         COMMAND ${MATLAB_BIN}/mex
                         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/mex/matlab/${matlab_target}.c -I${CMAKE_CURRENT_SOURCE_DIR}/src
                              ${Eep_sources}
                              ${MATLAB_MEX_CFLAGS}
                              -DLIBEEP_VERSION_MAJOR=${LIBEEP_VERSION_MAJOR}
                              -DLIBEEP_VERSION_MINOR=${LIBEEP_VERSION_MINOR}
                              -DLIBEEP_VERSION_PATCH=${LIBEEP_VERSION_PATCH}
                         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mex/matlab/${matlab_target}.c ${Eep_sources}
                        )
      add_custom_target(${matlab_target} ALL DEPENDS ${matlab_target}.${MATLAB_EXT})

      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${matlab_target}.${MATLAB_EXT} DESTINATION share/libeep/matlab)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mex/matlab/${matlab_target}.m)
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mex/matlab/${matlab_target}.m DESTINATION share/libeep/matlab)
    endif()
  endforeach()
elseif()
  message("please set the environment variable MATLAB to a matlab directry to build the MATLAB plugin")
endif()

##########
# python #
##########
add_subdirectory(python/v2)
add_subdirectory(python/v3)

if(UNIX)
  find_library(Math m)
  target_link_libraries(Eep ${Math})
endif(UNIX)

add_executable(demo_read_avr test/demo_read_avr.c)
add_executable(demo_read_cnt test/demo_read_cnt.c)
add_executable(demo_read_evt  test/demo_read_evt.c)
add_executable(demo_read_seg  test/demo_read_seg.c)
add_executable(demo_v4_read_cnt test/demo_v4_read_cnt.c)
add_executable(demo_v4_write_cnt test/demo_v4_write_cnt.c)
add_executable(demo_write_avr test/demo_write_avr.c)
add_executable(demo_write_cnt test/demo_write_cnt.c)
add_executable(libeep_info test/libeep_info.c)
target_link_libraries(demo_read_avr Eep ${MATH})
target_link_libraries(demo_read_cnt Eep ${MATH})
target_link_libraries(demo_read_evt Eep ${MATH})
target_link_libraries(demo_read_seg Eep ${MATH})
target_link_libraries(demo_v4_read_cnt Eep ${MATH})
target_link_libraries(demo_v4_write_cnt Eep ${MATH})
target_link_libraries(demo_write_avr Eep ${MATH})
target_link_libraries(demo_write_cnt Eep ${MATH})
target_link_libraries(libeep_info Eep)
