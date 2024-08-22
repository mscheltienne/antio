message("Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
# manylinux don't contain Development, use Development.Module or Development.SABIModule:
# https://github.com/pybind/pybind11/pull/2689/files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.SABIModule)
message("Python 3 include: ${Python3_INCLUDE_DIRS}")

set(Python_SOABI "abi3")
Python3_add_library(pyeep3 MODULE ./pyeep.c USE_SABI 3.9 WITH_SOABI)
set_target_properties(pyeep3 PROPERTIES OUTPUT_NAME pyeep)
target_link_libraries(pyeep3 PRIVATE EepStatic)
install(TARGETS pyeep3 DESTINATION share/libeep/python3/libeep)

if(UNIX)
  set_target_properties(pyeep3 PROPERTIES PREFIX "")
endif()
if(WIN32)
  set_target_properties(pyeep3 PROPERTIES SUFFIX .pyd)
endif()
