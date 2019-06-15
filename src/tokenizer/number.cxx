#include <apex/tokenizer.hxx>
#include <climits>

BEGIN_APEX_NAMESPACE

namespace tok {

result_t<unused_t> lexer_t::pp_number(range_t range) {
  result_t<unused_t> result;
  const char* begin = range.begin;

  // [lex.ppnumber]
  // pp-number:
  //   digit
  //   . digit
  range.advance_if('.');
  if(range.advance_if(isdigit)) {
    while(char c0 = toupper(range[0])) {
      char c1 = range[1];
      if(('E' == c0 || 'P' == c0) && ('+' == c1 || '-' == c1)) {
        // pp-number e sign
        // pp-number E sign
        // pp-number p sign
        // pp-number P sign
        range.begin += 2;
        continue;
      }

      if('\'' == c0 && (isalnum(c1) || '_' == c1)) {
        // pp-number ' digit
        // pp-number ' non-digit
        range.begin += 2;
        continue;
      }

      if('.' == c0) {
        // pp-number .
        ++range.begin;
        continue;
      }

      if(auto c = identifier_char(range, true)) {
        // pp-number digit
        // pp-number identifier-nondigit
        range.advance(c);
        continue;
      }

      break;
    }

    result = make_result(begin, range.begin, { });
  }

  return result;
}

struct floating_parts_t {
  range_t integer;        // digits before the .
  range_t fractional;     // digits after the .
  int exponent;           // after the exponent.
};

result_t<unused_t> lexer_t::decimal_sequence(range_t range) {
  const char* begin = range.begin;
  while(isdigit(range.peek()))
    ++range.begin;
  return make_result(begin, range.begin, { });
}

result_t<uint64_t> lexer_t::decimal_number(range_t range) {
  result_t<uint64_t> result;
  if(auto digits = decimal_sequence(range)) {
    range.advance(digits);

    uint64_t x = 0;
    for(const char* p = digits->range.begin; p != digits->range.end; ++p) {
      int y = *p - '0';
      uint64_t x2 = 10 * x + y;
      if(x2 < x)
        throw_error(p, "integer overflow in decimal literal");
      x = x2;
    }
    result = make_result(digits->range, x);
  }
  return result;
}

result_t<int> lexer_t::exponent_part(range_t range) {
  result_t<int> result;
  auto begin = range.begin;
  if(range.advance_if('e') || range.advance_if('E')) {
    bool sign = false;
    if(range.advance_if('-')) sign = true;
    else range.advance_if('+');

    // Expect a digit-sequence here.
    if(auto exp = decimal_number(range)) {
      range.advance(exp);
      if(exp->attr > INT_MAX)
        throw_error(exp->range.begin, "exponent is too large");

      int exponent = exp->attr;
      if(sign) exponent = -exponent;

      result = make_result(begin, range.begin, exponent);

    } else
      throw_error(range.begin, "expected digit-sequence in exponent-part");
  }
  return result;
}

result_t<long double> lexer_t::floating_point_literal(range_t range) {
  const char* begin = range.begin;
  floating_parts_t parts { };
  if(auto leading = decimal_sequence(range)) {
    range.advance(leading);
    parts.integer = leading->range;

    if(range.advance_if('.')) {
      // We've matched fractional-constant, so both the trailing digit-sequence
      // and exponent are optional.
      if(auto fractional = decimal_sequence(range)) {
        range.advance(fractional);
        parts.fractional = fractional->range;
      }

      if(auto exp = exponent_part(range)) {
        range.advance(exp);
        parts.exponent = exp->attr;
      }

    } else if(auto exp = exponent_part(range)) {
      range.advance(exp);
      parts.exponent = exp->attr;

    } else
      // A leading decimal sequence with no fraction or exp is an integer
      return { };

  } else if(range.advance_if('.')) {
    if(auto fractional = decimal_sequence(range)) {
      range.advance(fractional);
      parts.fractional = fractional->range;

      if(auto exp = exponent_part(range)) {
        range.advance(exp);
        parts.exponent = exp->attr;
      }
    }

  } else
    return { };
 
  // TODO: Assemble the floating-point literal by hand.
  // sscanf the floating point literal into long double.
  std::string s(begin, range.begin);
  long double x;
  sscanf(s.c_str(), "%Le", &x);

  return make_result(begin, range.begin, x);
}

result_t<uint64_t> lexer_t::integer_literal(range_t range) {
  result_t<uint64_t> result;

  // For now parse all numbers as base 10.
  if(auto number = decimal_number(range))
    result = number;
  
  return result;
};

result_t<token_t> lexer_t::number(range_t range) {
  result_t<token_t> result;
  if(auto num = pp_number(range)) {
    // The pp-number must be a floating-point-literal or integer-literal.
    range_t range = num->range;
    token_t token { };
    if(auto floating = floating_point_literal(range)) {
      range.advance(floating);

      result = make_result(range, token_t {
        tk_float,
        (int)tokenizer.floats.size(), 
        floating->range.begin, 
        floating->range.end
      });
      tokenizer.floats.push_back(floating->attr);

    } else if(auto integer = integer_literal(range)) {
      range.advance(integer);

      result = make_result(range, token_t {
        tk_int,
        (int)tokenizer.ints.size(), 
        integer->range.begin, 
        integer->range.end
      });
      tokenizer.ints.push_back(integer->attr);
    }

    if(range)
      throw_error(range.begin, "unexpected character in numeric literal");
  }
  return result;
}

} // namespace tok

END_APEX_NAMESPACE
