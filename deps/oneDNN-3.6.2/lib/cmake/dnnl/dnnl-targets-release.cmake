#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DNNL::dnnl" for configuration "Release"
set_property(TARGET DNNL::dnnl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(DNNL::dnnl PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/mkldnn.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mkldnn.dll"
  )

list(APPEND _cmake_import_check_targets DNNL::dnnl )
list(APPEND _cmake_import_check_files_for_DNNL::dnnl "${_IMPORT_PREFIX}/lib/mkldnn.lib" "${_IMPORT_PREFIX}/bin/mkldnn.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
