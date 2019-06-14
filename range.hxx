#pragma once
#include "util.hxx"
#include "tokens.hxx"

BEGIN_APEX_NAMESPACE



struct range_t {
  token_it begin, end;
  explicit operator bool() const { return begin < end; }

  token_t peek() const {
    return (begin < end) ? *begin : token_t { };
  }
  token_t next() {
    return (begin < end) ? *begin++ : token_t { };
  } 
  token_t advance_if(tk_kind_t kind) {
    return (begin < end && kind == begin->kind) ? *begin++ : token_t { };
  }

  void advance(token_it it) {
    begin = it;
  }
  void advance(range_t range) {
    begin = range.end;
  }
  template<typename type_t>
  void advance(const type_t& result) {
    if(result)
      advance(result->range.end);
  }
};

template<typename attr_t = unused_t>
using result_t = result_template_t<attr_t, range_t>;

template<typename attr_t = unused_t>
result_t<attr_t> make_result(range_t range, attr_t attr = { }) {
  return { range, std::move(attr) };
}

template<typename attr_t = unused_t>
result_t<attr_t> make_result(token_it begin, token_it end, attr_t attr = { }) {
  return make_result(range_t { begin, end }, std::move(attr));
}

END_APEX_NAMESPACE
