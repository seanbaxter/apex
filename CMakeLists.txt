cmake_minimum_required(VERSION 3.5.1)
project(apex)

set(CMAKE_CXX_STANDARD 17)
include_directories(include/apex)

set(SOURCE_FILES
  # apex.cxx
  src/util/utf.cxx

  src/parse/grammar.cxx

  src/tokenizer/tokens.cxx
  src/tokenizer/lexer.cxx  
  src/tokenizer/operators.cxx
  src/tokenizer/tokenizer.cxx  
  src/tokenizer/number.cxx

  src/autodiff/autodiff.cxx
)

add_library(apex SHARED
  ${SOURCE_FILES}
)
