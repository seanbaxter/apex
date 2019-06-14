cmake_minimum_required(VERSION 3.5.1)
project(apex)

set(CMAKE_CXX_STANDARD 17)


set(SOURCE_FILES
  apex.cxx
  grammar.cxx
  tokens.cxx
  lexer.cxx
  utf.cxx
  operators.cxx
  tokenizer.cxx
  number.cxx
)

add_executable(
  apex ${SOURCE_FILES}
)
