cmake_minimum_required(VERSION 3.20)

project(TwoDFL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MKL CONFIG REQUIRED)

add_executable(two_d_fl
    main.cpp
    src/subroutines.cpp
)

add_executable(subroutines_chk
    tests/subroutines_chk.cpp
    src/subroutines.cpp
)

target_include_directories(two_d_fl PRIVATE include)
target_include_directories(subroutines_chk PRIVATE include)

target_link_libraries(two_d_fl PRIVATE MKL::MKL)
target_link_libraries(subroutines_chk PRIVATE MKL::MKL)

if(MSVC)
    target_compile_options(two_d_fl PRIVATE /W4 /permissive-)
    target_compile_options(subroutines_chk PRIVATE /W4 /permissive-)
else()
    target_compile_options(two_d_fl PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(subroutines_chk PRIVATE -Wall -Wextra -Wpedantic)
endif()