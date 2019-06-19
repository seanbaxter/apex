#include <apex/value.hxx>

BEGIN_APEX_NAMESPACE

number_t number_t::to_boolean() const {
  return switch_all([](auto x) {
    return (bool)x;
  });
}

number_t number_t::to_integral() const {
  return switch_all([](auto x) {
    return (int64_t)x;
  });
}

number_t number_t::to_floating() const {
  return switch_all([](auto x) {
    return (double)x;
  });
}

number_t number_t::to_kind(number_kind_t kind2) const {
  number_t result { };
  switch(kind2) {
    case number_kind_bool:
      result = to_boolean();
      break;

    case number_kind_int:
      result = to_integral();
      break;

    case number_kind_float:
      result = to_floating();
      break;
  }
  return result;
}

std::string number_t::to_string() const {
  if(number_kind_bool == kind) {
    return b ? "true" : "false";

  } else {
    return switch_numeric([](auto x) {
      return std::to_string(x);
    });
  }
}

number_kind_t common_arithmetic_kind(number_kind_t left, number_kind_t right) {
  if(number_kind_float == left || number_kind_float == right)
    return number_kind_float;
  else
    return number_kind_int;
}

////////////////////////////////////////////////////////////////////////////////

number_t value_unary(expr_op_t op, number_t value) {
  number_t result { };
  switch(op) {
    case expr_op_inc_post:
    case expr_op_dec_post:
    case expr_op_inc_pre:
    case expr_op_dec_pre:
      break;

    case expr_op_complement:
      if(!value.is_floating()) 
        result = ~value.convert<int64_t>();
      break;

    case expr_op_negate:
      result = !value.convert<bool>();
      break;

    case expr_op_plus:
      result = value;
      break;
    
    case expr_op_minus:
      switch(value.kind) {
        case number_kind_bool:
          break;

        case number_kind_int:
          result = number_t(-value.i);
          break;

        case number_kind_float:
          result = number_t(-value.d);
          break;
      }
      break;
  }

  return result;
}

number_t value_binary(expr_op_t op, number_t left, number_t right) {
  number_t result { };
  
  switch(op) {
    case expr_op_add:
    case expr_op_sub:
    case expr_op_mul:
    case expr_op_div:
      // Promote to a common type.

    case expr_op_shl:
    case expr_op_shr:
    case expr_op_bit_and:
    case expr_op_bit_xor:
    case expr_op_bit_or:
      // Integer only.
      if(left.is_integral() && right.is_integral()) {
        int64_t x = 0;
        switch(op) {
          case expr_op_shl:     x = left.i<< right.i; break;
          case expr_op_shr:     x = left.i>> right.i; break;
          case expr_op_bit_and: x = left.i & right.i; break;
          case expr_op_bit_xor: x = left.i ^ right.i; break;
          case expr_op_bit_or:  x = left.i | right.i; break;
        }
        result = number_t(x);
      }
      break;

    case expr_op_lt:
    case expr_op_gt:
    case expr_op_lte:
    case expr_op_gte:
      // Integer or float.
      if(left.is_arithmetic() && right.is_arithmetic()) {
        bool x = false;
        
        if(left.is_floating() || right.is_floating()) {
          double a = left.convert<double>();
          double b = right.convert<double>();
          switch(op) {
            case expr_op_lt:  x = a <  b; break;
            case expr_op_gt:  x = a >  b; break;
            case expr_op_lte: x = a <= b; break;
            case expr_op_gte: x = a >= b; break;
          }

        } else {
          assert(left.is_integral() && right.is_integral());
          switch(op) {
            case expr_op_lt:  x = left.i <  right.i; break;
            case expr_op_gt:  x = left.i >  right.i; break;
            case expr_op_lte: x = left.i <= right.i; break;
            case expr_op_gte: x = left.i >= right.i; break;
          }
        }
        result = number_t(x);
      }
      break;

    case expr_op_eq:
    case expr_op_ne: {
      number_kind_t kind = common_arithmetic_kind(left.kind, right.kind);
      if(kind != left.kind) left = left.to_kind(kind);
      if(kind != right.kind) right = right.to_kind(kind);
      bool x = false;
      if(number_kind_bool == kind) {
        switch(op) {
          case expr_op_eq: x = left.b == right.b; break;
          case expr_op_ne: x = left.b != right.b; break;
        }

      } else if(number_kind_int == kind) {
        switch(op) {
          case expr_op_eq: x = left.i == right.i; break;
          case expr_op_ne: x = left.i != right.i; break;
        }

      } else {
        switch(op) {
          case expr_op_eq: x = left.d == right.d; break;
          case expr_op_ne: x = left.d != right.d; break;
        }
      }
      result = number_t(x);
    }
    break;

    case expr_op_log_and:
    case expr_op_log_or: {
      left = left.to_boolean();
      right = right.to_boolean();
      bool x = false;
      switch(op) {
        case expr_op_log_and: x = left.b && right.b; break;
        case expr_op_log_or:  x = left.b && right.b; break;
      }
      result = number_t(x);
      break;
    }

    case expr_op_sequence:
      result = right;
      break;
  }

  return result;
}


END_APEX_NAMESPACE
