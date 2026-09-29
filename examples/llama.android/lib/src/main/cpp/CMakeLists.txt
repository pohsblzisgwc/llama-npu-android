cmake_minimum_required(VERSION 3.22)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
set(JNILIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libllama.so)

add_library(llama-common SHARED IMPORTED)
set_target_properties(llama-common PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libllama-common.so)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libggml.so)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libggml-base.so)

add_library(mtmd SHARED IMPORTED)
set_target_properties(mtmd PROPERTIES IMPORTED_LOCATION ${JNILIBS_DIR}/libmtmd.so)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        ${LLAMA_SRC}/tools/mtmd)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        llama-common
        mtmd
        ggml
        ggml-base
        android
        log)
