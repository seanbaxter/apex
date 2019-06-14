#pragma once
#include "tokenizer.hxx"

BEGIN_APEX_NAMESPACE

enum expr_op_t : uint8_t {
  expr_op_none = 0,

  // postfix. 
  expr_op_inc_post,
  expr_op_dec_post, 

  // prefix.
  expr_op_inc_pre,          // ++x
  expr_op_dec_pre,          // --x
  expr_op_complement,       // ~x
  expr_op_negate,           // !x
  expr_op_plus,             // +x
  expr_op_minus,            // -x
  expr_op_addressof,        // &x
  expr_op_indirection,      // *x

  // Right-associative binary operators.
  expr_op_ptrmem_dot,
  expr_op_ptrmem_arrow,

  // Left-associative operations.
  expr_op_mul,
  expr_op_div,
  expr_op_mod,
  expr_op_add,
  expr_op_sub,
  expr_op_shl,
  expr_op_shr,
  expr_op_lt,
  expr_op_gt,
  expr_op_lte,
  expr_op_gte,
  expr_op_eq,
  expr_op_ne,
  expr_op_bit_and,
  expr_op_bit_xor,
  expr_op_bit_or,
  expr_op_log_and,
  expr_op_log_or,

  // Right-associative operations.  
  expr_op_assign,
  expr_op_assign_mul,
  expr_op_assign_div,
  expr_op_assign_mod,
  expr_op_assign_add,
  expr_op_assign_sub,
  expr_op_assign_shl,
  expr_op_assign_shr,
  expr_op_assign_and,
  expr_op_assign_or,
  expr_op_assign_xor,

  expr_op_ternary,
  expr_op_sequence,
};

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

struct node_t {
  enum kind_t {
    kind_ident,
    kind_unary,
    kind_binary,      // left-to-right evaluation
    kind_assign,      // right-to-left evaluation
    kind_ternary,
    kind_call,
    kind_char,
    kind_string,
    kind_int,
    kind_bool,
    kind_float,
    kind_subscript,
    kind_member,
    kind_braced,
  } kind;

  node_t(kind_t kind) : kind(kind) { }
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
  std::vector<token_t> tokens;
  node_ptr_t root;
};

parse_t parse_expression(const char* str);

////////////////////////////////////////////////////////////////////////////////

struct node_ident_t : node_t {
  node_ident_t() : node_t(kind_ident) { }
  static bool classof(const node_t* p) { return kind_ident == p->kind; }

  std::string s;
};

struct node_unary_t : node_t {
  node_unary_t() : node_t(kind_unary) { }
  static bool classof(const node_t* p) { return kind_unary == p->kind; }

  expr_op_t op;
  node_ptr_t a;
};

struct node_binary_t : node_t {
  node_binary_t() : node_t(kind_binary) { }
  static bool classof(const node_t* p) { return kind_binary == p->kind; }

  expr_op_t op;
  node_ptr_t a, b;
};

struct node_assign_t : node_t {
  node_assign_t() : node_t(kind_assign) { }
  static bool classof(const node_t* p) { return kind_assign == p->kind; }

  expr_op_t op;
  node_ptr_t a, b;
};

struct node_ternary_t : node_t {
  node_ternary_t() : node_t(kind_ternary) { }
  static bool classof(const node_t* p) { return kind_ternary == p->kind; }

  node_ptr_t a, b, c;
};

struct node_char_t : node_t {
  node_char_t(char32_t c) : node_t(kind_char), c(c) { }
  static bool classof(const node_t* p) { return kind_char == p->kind; }

  // UCS code for character. Caller should use UTF to_utf8 to convert back
  // to a UTF-8 string.
  char32_t c;
};

struct node_string_t : node_t {
  node_string_t(std::string s) : node_t(kind_string), s(std::move(s)) { }
  static bool classof(const node_t* p) { return kind_string == p->kind; }

  std::string s;
};

struct node_int_t : node_t {
  node_int_t(uint64_t ui) : node_t(kind_int), ui(ui) { }
  static bool classof(const node_t* p) { return kind_int == p->kind; }

  uint64_t ui;
};

struct node_bool_t : node_t {
  node_bool_t(bool b) : node_t(kind_bool), b(b) { }
  static bool classof(const node_t* p) { return kind_bool == p->kind; }

  bool b;
};

struct node_float_t : node_t {
  node_float_t(long double ld) : node_t(kind_float), ld(ld) { }
  static bool classof(const node_t* p) { return kind_float == p->kind; }

  long double ld;
};

struct node_call_t : node_t {
  node_call_t() : node_t(kind_call) { }
  static bool classof(const node_t* p) { return kind_call == p->kind; }

  node_ptr_t f;
  std::vector<node_ptr_t> args;
};

struct node_subscript_t : node_t {
  node_subscript_t() : node_t(kind_subscript) { }
  static bool classof(const node_t* p) { return kind_subscript == p->kind; }

  node_ptr_t lhs;
  std::vector<node_ptr_t> args;
};

struct node_member_t : node_t {
  node_member_t() : node_t(kind_member) { }
  static bool classof(const node_t* p) { return kind_member == p->kind; }

  tk_kind_t tk;           // dot or arrow
  node_ptr_t lhs;
  std::string member;
};

struct node_braced_t : node_t {
  node_braced_t() : node_t(kind_braced) { }
  static bool classof(const node_t* p) { return kind_braced == p->kind; }

  std::vector<node_ptr_t> args;
};

} // namespace parse


END_APEX_NAMESPACE
