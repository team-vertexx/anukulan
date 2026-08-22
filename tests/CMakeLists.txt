function(sankhya_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE sankhya)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE
    SANKHYA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

sankhya_test(test_sparse)
sankhya_test(test_mps)
sankhya_test(test_standard_form)
sankhya_test(test_scaling)
