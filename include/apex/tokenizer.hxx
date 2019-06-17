#pragma once
#include <apex/tokens.hxx>

BEGIN_APEX_NAMESPACE

namespace parse {

struct range_t;

}

namespace tok {

struct range_t {
  const char* begin, *end;
  explicit operator bool() const {
    return begin < end;
  }
  void advance(const char* p) {
    begin = p;
  }
  char advance_if(char c) {
    return (begin < end && *begin == c) ? *begin++ : 0;
  }
  template<typename func_t>
  char advance_if(func_t f) {
    return (begin < end && f(*begin)) ? *begin++ : 0;
  }
  template<typename type_t>
  void advance(const type_t& result) {
    if(result)
      advance(result->range.end);
  }

  char operator[](ptrdiff_t index) {
    return (begin + index < end) ? begin[index] : 0;
  }

  char peek() const {
    return (begin < end) ? *begin : 0; 
  }
  char next() {
    return (begin < end) ? *begin++ : 0;
  }

  bool match(const char* s) const {
    const char* p = begin;
    while(*s && p < end && *p++ == *s) ++s;
    return !*s;
  }

  bool match_advance(const char* s) {
    const char* p = begin;
    while(*s && p < end && *p++ == *s) ++s;
    bool success = !*s;
    if(success) begin = p;
    return success;
  }
};

template<typename attr_t>
using result_t = result_template_t<attr_t, range_t>;

template<typename attr_t = unused_t>
result_t<attr_t> make_result(range_t range, attr_t attr = { }) {
  return { range, std::move(attr) };
}

template<typename attr_t = unused_t>
result_t<attr_t> make_result(const char* begin, const char* end, 
  attr_t attr = { }) {
  return make_result(range_t { begin, end }, std::move(attr));
}

// operators.cxx. Match the longest operator.
result_t<tk_kind_t> match_operator(range_t range);

struct tokenizer_t;

struct lexer_t {
  lexer_t(tokenizer_t& tokenizer) : tokenizer(tokenizer) { }

  result_t<token_t> char_literal(range_t range);
  result_t<char32_t> c_char(range_t range);
  
  result_t<token_t> string_literal(range_t range);
  result_t<char32_t> s_char(range_t range);

  // Match a-zA-Z or a UCS. If digit is true, also match a digit.
  result_t<char32_t> identifier_char(range_t range, bool digit);

  // Read an extended character.
  result_t<char32_t> ucs(range_t range);

  // Read a character sequence matching any number.
  // This conforms to the C++17 definition pp-number.
  result_t<unused_t> pp_number(range_t range);
  result_t<unused_t> decimal_sequence(range_t range);
  result_t<uint64_t> decimal_number(range_t range);
  result_t<int> exponent_part(range_t range);

  result_t<uint64_t> integer_literal(range_t range);
  result_t<long double> floating_point_literal(range_t range);
  result_t<token_t> number(range_t range);

  result_t<token_t> literal(range_t range);
  result_t<token_t> identifier(range_t range);
  result_t<token_t> operator_(range_t range);
  result_t<token_t> token(range_t range);

  const char* skip_comment(range_t range);
  bool advance_skip(range_t& range);


  void throw_error(const char* pos, const char* msg);

  tokenizer_t& tokenizer;
};

struct tokenizer_t {
  std::vector<std::string> strings;
  std::vector<uint64_t> ints;
  std::vector<long double> floats;

  // Byte offset for each line start.
  std::vector<int> line_offsets;

  // Original text we tokenized.
  std::string text;

  // The text divided into tokens.
  std::vector<token_t> tokens;

  parse::range_t token_range() const;

  int reg_string(range_t range);
  int find_string(range_t range) const;

  // Return 0-indexed line and column offsets for the token at
  // the specified byte offset. This performs UCS decoding to support
  // multibyte characters.
  int token_offset(source_loc_t loc) const;
  int token_line(int offset) const;
  int token_col(int offset, int line) const;
  std::pair<int, int> token_linecol(int offset) const;
  std::pair<int, int> token_linecol(source_loc_t loc) const;
 
  void tokenize();
};

} // namespace tok

END_APEX_NAMESPACE
