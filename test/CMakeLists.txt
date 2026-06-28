add_executable(test_reorder test_reorder.cpp)
target_link_libraries(test_reorder ${PROJECT_NAME})

add_executable(test_compress test_compress.cpp)
target_link_libraries(test_compress ${PROJECT_NAME} ${METIS_LIBRARY} GSL::gsl SZ3::SZ3)

add_executable(test_decompress test_decompress.cpp)
target_link_libraries(test_decompress ${PROJECT_NAME} ${METIS_LIBRARY} GSL::gsl SZ3::SZ3)

add_executable(test_comp_decomp_cp_preserve test_comp_decomp_cp.cpp)
target_link_libraries(test_comp_decomp_cp_preserve ${PROJECT_NAME} ${METIS_LIBRARY} SZ3::SZ3)

# add_executable(cp_extraction_3d_unstructured cp_extraction_3d_unstructured.cpp)
# target_link_libraries (cp_extraction_3d_unstructured ${PROJECT_NAME} FTK::libftk)

add_executable(test_parameterization_tsunami test_parameterization_tsunami.cpp)
target_link_libraries(test_parameterization_tsunami ${PROJECT_NAME})

add_executable(test_parameterization_tsunami_multilevel test_parameterization_tsunami_multilevel.cpp)
target_link_libraries(test_parameterization_tsunami_multilevel ${PROJECT_NAME})

add_executable(test_compress_tet test_compress_tet.cpp)
target_link_libraries(test_compress_tet ${PROJECT_NAME})
add_executable(test_decompress_tet test_decompress_tet.cpp)
target_link_libraries(test_decompress_tet ${PROJECT_NAME})

add_executable(test_compress_barycentric test_compress_barycentric.cpp)
target_link_libraries(test_compress_barycentric ${PROJECT_NAME})

add_executable(test_decompress_barycentric test_decompress_barycentric.cpp)
target_link_libraries(test_decompress_barycentric ${PROJECT_NAME})