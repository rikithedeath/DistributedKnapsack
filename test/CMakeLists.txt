find_package(GTest REQUIRED CONFIG)
set(TARGET_NAME distributed_knapsack)
set(UT_TARGET_NAME "distributed_knapsack_tests")
file(GLOB TEST_SOURCES  *.cpp *c )
get_target_property(TGT_INCLUDE_DIRS ${TARGET_NAME} INCLUDE_DIRECTORIES)
get_target_property(TGT_LIBRARIES ${TARGET_NAME} LINK_LIBRARIES)
get_target_property(SRC_LIST ${TARGET_NAME} SOURCES)
# Match the file name only: a regex on the whole path (".*main.*") also dropped
# every source when the checkout lived under e.g. "DistributedKnapsack-main/".
list(FILTER SRC_LIST EXCLUDE REGEX "/main\\.cpp$")
list(FILTER SRC_LIST EXCLUDE REGEX "/lib\\.cpp$")
add_executable(${UT_TARGET_NAME} ${TEST_SOURCES} ${SRC_LIST})
target_sources(${UT_TARGET_NAME} PUBLIC
    FILE_SET CXX_MODULES
    TYPE CXX_MODULES
    BASE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
    FILES ${KP_MODULES}
)
target_include_directories(${UT_TARGET_NAME} PUBLIC ${TGT_INCLUDE_DIRS})
target_link_libraries(${UT_TARGET_NAME} PUBLIC ${TGT_LIBRARIES} PRIVATE GTest::gtest GTest::gtest_main  fmt::fmt)

add_test(
    NAME python_hello_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hello.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
set_tests_properties(python_hello_test PROPERTIES ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:distributed_knapsack>")

if(KP_MPI)
    add_subdirectory(${PROJECT_SOURCE_DIR}/mpi/test ${CMAKE_BINARY_DIR}/test/mpi)
endif()

