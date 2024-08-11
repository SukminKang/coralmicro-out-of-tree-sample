cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

project(coralmicro-app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(coralmicro)
add_subdirectory(coralmicro)

add_executable_m7(coralmicro-app
        main.cc
        DATA
        ${PROJECT_SOURCE_DIR}/models/multi/inception_v2/5_1_edgetpu.tflite
        ${PROJECT_SOURCE_DIR}/models/multi/inception_v2/5_2_edgetpu.tflite
        ${PROJECT_SOURCE_DIR}/models/multi/inception_v2/5_3_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/mobilenet_v2_1.0_224_quant_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/mobilenet_v2_1.0_224_quant.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/01.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/02.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/03.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/04.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/05.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/06.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/07.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/08.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/09.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/10.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/11.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/12.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/13.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/14.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/15.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/16.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/17.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/18.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/19.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/20.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/21.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/22.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/23.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/24.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/25.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/26.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/27.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/28.tflite
        # ${PROJECT_SOURCE_DIR}/models/cpu/29.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/01_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/02_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/03_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/04_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/05_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/06_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/07_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/08_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/09_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/10_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/11_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/12_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/13_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/14_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/15_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/16_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/17_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/18_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/19_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/20_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/21_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/22_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/23_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/24_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/25_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/26_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/27_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/28_edgetpu.tflite
        # ${PROJECT_SOURCE_DIR}/models/tpu/29_edgetpu.tflite

)

target_link_libraries(coralmicro-app
        libs_base-m7_freertos
)
