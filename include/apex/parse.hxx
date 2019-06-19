#pragma once
#include <apex/tokenizer.hxx>
#include <apex/value.hxx>

BEGIN_APEX_NAMESPACE

namespace parse {

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

struct parse_exception_t : std::runtime_error {
  parse_exception_t(const std::string& err) : std::runtime_error(err) { }
};

struct node_t {
  enum kind_t {
    kind_ident,
    kind_unary,
    kind_binary,
    kind_assign,
    kind_ternary,
    kind_call,
    kind_char,
    kind_string,
    kind_number,
    kind_bool,
    kind_subscript,
    kind_member,
    kind_braced,
  };

  kind_t kind;
  source_loc_t loc;

  node_t(kind_t kind, source_loc_t loc) : kind(kind), loc(loc) { }
  virtual ~node_t() { }

  template<typename derived_t>
  derived_t* as() {
    return derived_t::classof(this) ? 
      static_cast<derived_t*>(this) : 
      nullptr;
  }

  template<typename derived_t>
  const derived_t* as() const {
    return derived_t::classof(this) ? 
      static_cast<const derived_t*>(this) : 
      nullptr;
  }
};
typedef std::unique_ptr<node_t> node_ptr_t;
typedef std::vector<node_ptr_t> node_list_t;

struct parse_t {
  tok::tokenizer_t tokenizer;
  node_ptr_t root;
};

parse_t parse_expression(const char* str);

////////////////////////////////////////////////////////////////////////////////

struct node_ident_t : node_t {
  node_ident_t(source_loc_t loc) : node_t(kind_ident, loc) { }
  static bool classof(const node_t* p) { return kind_ident == p->kind; }

  std::string s;
};

struct node_unary_t : node_t {
  node_unary_t(source_loc_t loc) : node_t(kind_unary, loc) { }
  static bool classof(const node_t* p) { return kind_unary == p->kind; }

  expr_op_t op;
  node_ptr_t a;
};

struct node_binary_t : node_t {
  node_binary_t(source_loc_t loc) : node_t(kind_binary, loc) { }
  static bool classof(const node_t* p) { return kind_binary == p->kind; }

  expr_op_t op;
  node_ptr_t a, b;
};

struct node_assign_t : node_t {
  node_assign_t(source_loc_t loc) : node_t(kind_assign, loc) { }
  static bool classof(const node_t* p) { return kind_assign == p->kind; }

  expr_op_t op;
  node_ptr_t a, b;
};

struct node_ternary_t : node_t {
  node_ternary_t(source_loc_t loc) : node_t(kind_ternary, loc) { }
  static bool classof(const node_t* p) { return kind_ternary == p->kind; }

  node_ptr_t a, b, c;
};

struct node_char_t : node_t {
  node_char_t(char32_t c, source_loc_t loc) : node_t(kind_char, loc), c(c) { }
  static bool classof(const node_t* p) { return kind_char == p->kind; }

  // UCS code for character. Caller should use UTF to_utf8 to convert back
  // to a UTF-8 string.
  char32_t c;
};

struct node_string_t : node_t {
  node_string_t(std::string s, source_loc_t loc) : 
    node_t(kind_string, loc), s(std::move(s)) { }
  static bool classof(const node_t* p) { return kind_string == p->kind; }

  std::string s;
};

struct node_bool_t : node_t {
  node_bool_t(bool b, source_loc_t loc) : node_t(kind_bool, loc), b(b) { }
  static bool classof(const node_t* p) { return kind_bool == p->kind; }

  bool b;
};

struct node_number_t : node_t {
  node_number_t(number_t number, source_loc_t loc) :
    node_t(kind_number, loc), x(number) { }
  static bool classof(const node_t* p) { return kind_number == p->kind; }

  number_t x;
};

struct node_call_t : node_t {
  node_call_t(source_loc_t loc) : node_t(kind_call, loc) { }
  static bool classof(const node_t* p) { return kind_call == p->kind; }

  node_ptr_t f;
  std::vector<node_ptr_t> args;
};

struct node_subscript_t : node_t {
  node_subscript_t(source_loc_t loc) : node_t(kind_subscript, loc) { }
  static bool classof(const node_t* p) { return kind_subscript == p->kind; }

  node_ptr_t lhs;
  std::vector<node_ptr_t> args;
};

struct node_member_t : node_t {
  node_member_t(source_loc_t loc) : node_t(kind_member, loc) { }
  static bool classof(const node_t* p) { return kind_member == p->kind; }

  tk_kind_t tk;           // dot or arrow
  node_ptr_t lhs;
  std::string member;
};

struct node_braced_t : node_t {
  node_braced_t(source_loc_t loc) : node_t(kind_braced, loc) { }
  static bool classof(const node_t* p) { return kind_braced == p->kind; }

  std::vector<node_ptr_t> args;
};

} // namespace parse


END_APEX_NAMESPACE
