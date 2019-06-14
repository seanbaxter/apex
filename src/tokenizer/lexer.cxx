#include "tokenizer.hxx"
#include <cctype>

BEGIN_APEX_NAMESPACE

namespace tok {

result_t<token_t> lexer_t::char_literal(range_t range) {
  const char* begin = range.begin;
  result_t<token_t> result;

  if(range.advance_if('\'')) {
    char32_t char_;
    if(auto c = c_char(range)) {
      char_ = c->attr;

    } else
      throw_error(range.begin, "expected character in literal");

    if(!range.advance_if('\''))
      throw_error(range.begin, "expected \"'\" to end character literal");

    result = make_result(begin, range.begin, token_t {
      tk_char, (int)char_, begin, range.begin
    });
  }
  return result;
}

result_t<char32_t> lexer_t::c_char(range_t range) {
  // Any character except '.
  // if(range[0] == '')
  return { };
}

result_t<token_t> lexer_t::string_literal(range_t range) {
  return { };
} 

result_t<char32_t> lexer_t::s_char(range_t range) {
  // Any character except ".
  return { };
}

result_t<char32_t> lexer_t::identifier_char(range_t range, bool digit) {
  const char* begin = range.begin;
  result_t<char32_t> result;

  if(char c = range.next()) {
    bool c1 = digit && isdigit(c);
    if(c1 || isalpha(c) || '_' == c) 
      result = make_result(begin, range.begin, (char32_t)c);
    else
      result = ucs(range);
  }
  return result;
}

result_t<char32_t> lexer_t::ucs(range_t range) {
  const char* begin = range.begin;
  result_t<char32_t> result;

  if(range && (0x80 & range.begin[0])) {
    std::pair<int, int> p = from_utf8(range.begin);
    range.begin += p.first;
    result = make_result(begin, range.begin, (char32_t)p.second);
  }
  return result;
}

const char* lexer_t::skip_comment(range_t range) {
  while(true) {
    // Eat the blank characters.
    while(range.advance_if(isblank));

    const char* begin = range.begin;
    if(range.match_advance("//")) {
      // Match a C++-style comment.
      auto f = [](char c) {
        return '\n' != c;
      };

      while(range.advance_if(f));

    } else if(range.match_advance("/*")) {
      // Match a C-style comment.
      while(!range.match("*/") && range.next());

      if(!range.match_advance("*/"))
        throw_error(begin, "unterminated C-style comment: expected */");

    } else
      break;
  }

  return range.begin;
}

result_t<token_t> lexer_t::literal(range_t range) {
  result_t<token_t> result = number(range);
  if(!result) result = char_literal(range);
  if(!result) result = string_literal(range);
  return result;
}

result_t<token_t> lexer_t::operator_(range_t range) {
  result_t<token_t> result;
  if(auto match = match_operator(range)) {
    token_t token { match->attr, 0, match->range.begin, match->range.end };
    return make_result(match->range, token);
  }
  return result;
}

result_t<token_t> lexer_t::identifier(range_t range) {
  const char* begin = range.begin;
  result_t<token_t> result;

  if(auto c = identifier_char(range, false)) {
    range.advance(c);

    while(auto c = identifier_char(range, true))
      range.advance(c);

    int ident = tokenizer.reg_string(range_t { begin, range.begin });
    token_t token { tk_ident, ident, begin, range.begin };
    result = make_result(begin, range.begin, token);
  }
  return result;
}

result_t<token_t> lexer_t::token(range_t range) {
  result_t<token_t> result = literal(range);
  if(!result) result = identifier(range);
  if(!result) result = operator_(range);
  return result;
}


bool lexer_t::advance_skip(range_t& range) {
  const char* next = skip_comment(range);
  bool advance = next != range.begin;
  range.begin = next;
  return advance;
}

void lexer_t::throw_error(const char* p, const char* msg) {
  printf("Error thrown %s\n", msg);
  exit(0);
}

} // namespace tok

END_APEX_NAMESPACE
