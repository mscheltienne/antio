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

add_subdirectory(python)
