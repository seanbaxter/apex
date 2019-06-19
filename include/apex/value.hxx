#pragma once
#include <apex/util.hxx>

BEGIN_APEX_NAMESPACE

enum number_kind_t {
  number_kind_none,
  number_kind_bool,
  number_kind_int,
  number_kind_float
};

struct number_t {
  number_kind_t kind;
  union {
    bool b;
    int64_t i;
    double d;
  };

  number_t() : kind(number_kind_none) { }
  number_t(bool b) : kind(number_kind_bool), b(b) { }
  number_t(int64_t i) : kind(number_kind_int), i(i) { }
  number_t(double d) : kind(number_kind_float), d(d) { }

  bool is_boolean() const { return number_kind_bool == kind; }
  bool is_integral() const { return number_kind_int == kind; }
  bool is_floating() const { return number_kind_float == kind; }
  bool is_arithmetic() const { return is_integral() || is_floating(); }

  explicit operator bool() const { 
    return number_kind_none != kind;
  }

  number_t to_boolean() const;
  number_t to_integral() const;
  number_t to_floating() const;

  number_t to_kind(number_kind_t kind2) const;

  std::string to_string() const;


  template<typename func_t>
  auto switch_numeric(func_t f) const {
    switch(kind) {
      case number_kind_int:
        return f(i);

      case number_kind_float:
        return f(d);
    }
  }

  template<typename func_t>
  auto switch_all(func_t f) const {
    switch(kind) {
      case number_kind_bool:
        return f(b); 

      case number_kind_int:
        return f(i);

      case number_kind_float:
        return f(d);
    }
  }

  template<typename type_t>
  type_t convert() const {
    return switch_all([](auto x) { return (type_t)x; });
  }
};

number_kind_t common_arithmetic_kind(number_kind_t left, number_kind_t right);

////////////////////////////////////////////////////////////////////////////////

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

extern const char* expr_op_names[];

number_t value_unary(expr_op_t op, number_t value);
number_t value_binary(expr_op_t op, number_t left, number_t right);

END_APEX_NAMESPACE
