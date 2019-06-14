#include "parse.hxx"
#include "range.hxx"
#include <stack>
#include <cstring>

BEGIN_APEX_NAMESPACE

struct grammar_t {
  token_it advance_brace(range_t range);
  token_it advance_paren(range_t range);
  token_it advance_bracket(range_t range);

  result_t<range_t> parse_paren(range_t range);
  result_t<range_t> parse_brace(range_t range);
  result_t<range_t> parse_bracket(range_t range);

  template<typename F>
  auto parse_switch(range_t range, F f) {
    result_t<decltype(f((tk_kind_t)0))> result;
    if(auto x = f(range.peek()))
      result = make_result(range.begin, range.begin + 1, x);
    return result;
  }

  result_t<node_ptr_t> entity(range_t range, bool expect);
  result_t<node_ptr_t> literal(range_t range);
  result_t<node_ptr_t> primary_expression(range_t range, bool expect);
  result_t<node_ptr_t> expression(range_t range, bool expect);
  result_t<node_ptr_t> postfix_expression(range_t range, bool expect);
  result_t<node_ptr_t> postfix_operator(range_t range, node_ptr_t& node);
  result_t<node_list_t> paren_initializer(range_t range);
  result_t<node_ptr_t> unary_expression(range_t range, bool expect);
  result_t<node_ptr_t> binary_expression(range_t range, bool expect);
  result_t<node_ptr_t> logical_and_expression(range_t range, bool expect);
  result_t<node_ptr_t> logical_or_expression(range_t range, bool expect);
  result_t<node_ptr_t> assignment_expression(range_t range, bool expect);

  result_t<node_ptr_t> paren_expression(range_t range);
  result_t<node_ptr_t> braced_init_list(range_t range);
  result_t<node_ptr_t> initializer_clause(range_t range, bool expect);
  result_t<node_list_t> init_list(range_t range);

  void throw_error(token_it pos, const char* msg);
  void unexpected_token(token_it pos, const char* msg);
};

////////////////////////////////////////////////////////////////////////////////

token_it grammar_t::advance_brace(range_t range) {
  int count = 1;
  while(token_t token = range.next()) {
    if(tk_sym_paren_l == token)
      range.begin = advance_paren(range);
    else if(tk_sym_paren_r == token) 
      throw_error(range.begin - 1, "unbalanced ')' in brace set { }");

    else if(tk_sym_bracket_l == token)
      range.begin = advance_bracket(range);
    else if(tk_sym_bracket_r == token)
      throw_error(range.begin - 1, "unbalanced ']' in brace set { }");

    else if(tk_sym_brace_l == token)
      ++count;
    else if(tk_sym_brace_r == token) 
      --count;

    if(!count) break;
  }

  if(count)
    throw_error(range.begin, "no closing '}' in brace set { }");

  return range.begin;
}

token_it grammar_t::advance_paren(range_t range) {
  int count = 1;
  while(token_t token = range.next()) {
    if(tk_sym_bracket_l == token)
      range.begin = advance_bracket(range);
    else if(tk_sym_bracket_r == token)
      throw_error(range.begin - 1, "unbalanced ']' in paren set ( )");

    else if(tk_sym_brace_l == token)
      range.begin = advance_brace(range);
    else if(tk_sym_brace_r == token)
      throw_error(range.begin - 1, "unbalanced '}' in paren set ( )");

    else if(tk_sym_paren_l == token)
      ++count;
    else if(tk_sym_paren_r == token)
      --count;

    if(!count) break;
  }

  if(count)
    throw_error(range.begin, "no closing ')' in paren set ( )");

  return range.begin;
}

token_it grammar_t::advance_bracket(range_t range) {
  int count = 1;
  while(token_t token = range.next()) {
    if(tk_sym_brace_l == token)
      range.begin = advance_brace(range);
    else if(tk_sym_brace_r == token)
      throw_error(range.begin - 1, "unbalanced '}' in bracket set [ ]");

    else if(tk_sym_paren_l == token)
      range.begin = advance_paren(range);
    else if(tk_sym_paren_r == token) 
      throw_error(range.begin - 1, "unbalanced ')' in bracket set [ ]");

    else if(tk_sym_bracket_l == token )
      ++count;
    else if(tk_sym_bracket_r == token)
      --count;

    if(!count) break;
  }

  if(count)
    throw_error(range.begin, "no closing ']' in bracket set [ ]");

  return range.begin;
}

result_t<range_t> grammar_t::parse_brace(range_t range) {
  result_t<range_t> result;
  token_it begin = range.begin;
  if(range.advance_if(tk_sym_brace_l)) {
    token_it end = advance_brace(range);
    result = make_result(begin, end, range_t { range.begin, end - 1 });
  }
  return result;
}

result_t<range_t> grammar_t::parse_paren(range_t range) {
  result_t<range_t> result;
  token_it begin = range.begin;
  if(range.advance_if(tk_sym_paren_l)) {
    token_it end = advance_paren(range);
    result = make_result(begin, end, range_t { range.begin, end - 1 });
  }
  return result;
}

result_t<range_t> grammar_t::parse_bracket(range_t range) {
  result_t<range_t> result;
  token_it begin = range.begin;
  if(range.advance_if(tk_sym_bracket_l)) {
    token_it end = advance_bracket(range);
    result = make_result(begin, end, range_t { range.begin, end - 1 });
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////

result_t<node_ptr_t> grammar_t::entity(range_t range, bool expect) {
  result_t<node_ptr_t> result;
  token_it begin = range.begin;
  if(token_t token = range.advance_if(tk_ident)) {
    auto ident = std::make_unique<node_ident_t>();
    result = make_result(begin, range.begin, std::move(ident));

  } else if(expect)
    throw_error(range.begin, "expected entity in expression");

  return result;
}

result_t<node_ptr_t> grammar_t::literal(range_t range) {
  token_it begin = range.begin;
  node_ptr_t node;
  switch(token_t token = range.next()) {

    case tk_int:


    case tk_float:
      break;

    case tk_char:


    case tk_string:

    case tk_kw_false:
      node = std::make_unique<node_bool_t>(false);
      break;

    case tk_kw_true:
      node = std::make_unique<node_bool_t>(true);
      break;

    default:
      break;
  }

  return make_result(begin, range.begin, std::move(node));
}

result_t<node_ptr_t> grammar_t::primary_expression(range_t range, bool expect) {
  result_t<node_ptr_t> result;

  switch(range.peek()) { 
    case tk_kw_false:
    case tk_kw_true:
    case tk_int:
    case tk_float:
    case tk_char:
    case tk_string:
      result = literal(range);
      break;

    case tk_sym_paren_l:
      result = paren_expression(range);
      break;

    default:
      result = entity(range, expect);
      break;
  }

  return result;
}

result_t<node_ptr_t> grammar_t::postfix_expression(range_t range, bool expect) {
  token_it begin = range.begin;
  result_t<node_ptr_t> result;
  if(auto primary = primary_expression(range, expect)) {
    range.advance(primary);
    node_ptr_t node = std::move(primary->attr);

    // Consume postfix operators until there are no more.
    while(auto op = postfix_operator(range, node)) {
      range.advance(op);
      assert(op->attr);
      node = std::move(op->attr);
    }

    assert(node);
    result = make_result(begin, range.begin, std::move(node));
  }

  return result;
}

result_t<node_ptr_t> grammar_t::postfix_operator(range_t range, 
  node_ptr_t& node) {

  token_it begin = range.begin;
  switch(token_t token = range.next()) {
    case tk_sym_minusminus: {
    case tk_sym_plusplus:
      auto unary = std::make_unique<node_unary_t>();
      unary->op = tk_sym_plusplus == token ? 
        expr_op_inc_post : expr_op_dec_post;
      unary->a = std::move(node);
      node = std::move(unary);
      break;
    }

    case tk_sym_bracket_l: {
      // Subscript operation.
      break;
    }

    case tk_sym_paren_l: {
      --range.begin;
      auto paren = paren_initializer(range);
      range.advance(paren);

      auto call = std::make_unique<node_call_t>();
      call->f = std::move(node);
      call->args = std::move(paren->attr);
      node = std::move(call);
      break;
    }

    case tk_sym_arrow:
    case tk_sym_dot: {

    }

    default:
      // We don't match any of the postfix expressions, so break the loop and
      // return to the caller.
    return { };
  }

  return make_result(begin, range.begin, std::move(node));
}

////////////////////////////////////////////////////////////////////////////////

expr_op_t switch_unary(tk_kind_t kind) {
  expr_op_t op = expr_op_none;
  switch(kind) {
    case tk_sym_plusplus:   op = expr_op_inc_pre;     break;
    case tk_sym_minusminus: op = expr_op_dec_pre;     break;
    case tk_sym_tilde:      op = expr_op_complement;  break;
    case tk_sym_bang:       op = expr_op_negate;      break;
    case tk_sym_plus:       op = expr_op_plus;        break;
    case tk_sym_minus:      op = expr_op_minus;       break;
    case tk_sym_amp:        op = expr_op_addressof;   break;
    case tk_sym_star:       op = expr_op_indirection; break;
    default:                                          break;
  }
  return op;
}

result_t<node_ptr_t> grammar_t::unary_expression(range_t range, bool expect) {
  token_it begin = range.begin;
  result_t<node_ptr_t> result;

  if(auto op = parse_switch(range, switch_unary)) {
    range.advance(op);
      
    auto rhs = unary_expression(range, true);
    range.advance(rhs);

    auto unary = std::make_unique<node_unary_t>();
    unary->op = op->attr;
    unary->a = std::move(rhs->attr);
    result = make_result(begin, range.begin, std::move(unary));

  } else 
    result = postfix_expression(range, expect);
  
  return result;
}

////////////////////////////////////////////////////////////////////////////////

enum ast_prec_t : uint8_t {
  // lowest precedence.
  ast_prec_any = 0,
  ast_prec_comma,
  ast_prec_assign,
  ast_prec_log_or,
  ast_prec_log_and,
  ast_prec_bit_or,
  ast_prec_bit_xor,
  ast_prec_bit_and,
  ast_prec_eq,
  ast_prec_cmp,
  ast_prec_shift,
  ast_prec_add,
  ast_prec_mul,
  ast_prec_ptr_to_mem,
  // highest precedence.
};

struct binary_desc_t {
  expr_op_t op;
  ast_prec_t prec;

  explicit operator bool() const { return op; }
};

binary_desc_t switch_binary(tk_kind_t kind) {
  binary_desc_t desc { };
  switch(kind) {
    // binary ->* and .*
    case tk_sym_arrowstar:  desc = { expr_op_ptrmem_arrow, ast_prec_ptr_to_mem }; break;
    case tk_sym_dotstar:    desc = { expr_op_ptrmem_dot,   ast_prec_ptr_to_mem }; break;
  
    // binary *, /, % with the same precedence.
    case tk_sym_star:       desc = { expr_op_mul,     ast_prec_mul        }; break;
    case tk_sym_slash:      desc = { expr_op_div,     ast_prec_mul        }; break;
    case tk_sym_percent:    desc = { expr_op_mod,     ast_prec_mul        }; break;
          
    // binary + and - with the same precedence.      
    case tk_sym_plus:       desc = { expr_op_add,     ast_prec_add        }; break;
    case tk_sym_minus:      desc = { expr_op_sub,     ast_prec_add        }; break;
          
    // <<, >> with the same precedence.      
    case tk_sym_ltlt:       desc = { expr_op_shl,     ast_prec_shift      }; break;
    case tk_sym_gtgt:       desc = { expr_op_shr,     ast_prec_shift      }; break;
          
    // <, >, <=, >= with the same precedence.      
    case tk_sym_lt:         desc = { expr_op_lt,      ast_prec_cmp        }; break;
    case tk_sym_gt:         desc = { expr_op_gt,      ast_prec_cmp        }; break;
    case tk_sym_lteq:       desc = { expr_op_lte,     ast_prec_cmp        }; break;
    case tk_sym_gteq:       desc = { expr_op_gte,     ast_prec_cmp        }; break;
          
    // == and != with the same precedence.      
    case tk_sym_eqeq:       desc = { expr_op_eq,      ast_prec_eq         }; break;
    case tk_sym_bangeq:     desc = { expr_op_ne,      ast_prec_eq         }; break;
        
    // bitwise AND &
    case tk_sym_amp:        desc = { expr_op_bit_and, ast_prec_bit_and    }; break;
  
    // bitwise XOR ^
    case tk_sym_caret:      desc = { expr_op_bit_xor, ast_prec_bit_xor    }; break;
  
    // bitwise OR |
    case tk_sym_pipe:       desc = { expr_op_bit_or,  ast_prec_bit_or     }; break;

    default:                                                                 break;
  }
  return desc;
}

struct item_t {
  node_ptr_t node;
  source_loc_t loc;
  binary_desc_t desc;
};

result_t<node_ptr_t> grammar_t::binary_expression(range_t range, bool expect) {

  std::vector<item_t> stack;
  auto fold = [&]() {
    while(stack.size() >= 2) {
      size_t size = stack.size();
      auto& lhs = stack[size - 2];
      auto& rhs = stack[size - 1];

      if(lhs.desc.prec >= rhs.desc.prec) {
        // Fold the two right-most expressions together.
        auto binary = std::make_unique<node_binary_t>();
        binary->op = lhs.desc.op;
        binary->a = std::move(lhs.node);
        binary->b = std::move(rhs.node);
        lhs.node = std::move(binary);

        // Use the descriptor for the rhs for this subexpression.
        lhs.loc = rhs.loc;
        lhs.desc = rhs.desc;

        // Pop the rhs.
        stack.pop_back();
      } else
        break;
    }
  };

  token_it begin = range.begin;
  result_t<node_ptr_t> result;
  if(auto lhs = unary_expression(range, false)) {
    range.advance(lhs);
    stack.push_back({ std::move(lhs->attr) });

    while(true) {
      item_t& item = stack.back();
      if(auto op = parse_switch(range, switch_binary)) {
        range.advance(op);
        item.desc = op->attr;
      } else
        // No operator found. This is the end of the binary-expression.
        break;

      // Fold the expressions to the left with equal or lesser precedence.
      fold();

      // Read the next expression.
      auto rhs = unary_expression(range, true);
      assert(rhs);
      range.advance(rhs);
      stack.push_back({ std::move(rhs->attr) });
      assert(stack.back().node);
    }

    // Fold all the remaining expressions.
    stack.back().desc.prec = ast_prec_any;
    fold();

    assert(1 == stack.size());
    result = make_result(begin, range.begin, std::move(stack[0].node));
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////

result_t<node_ptr_t> grammar_t::logical_and_expression(range_t range, 
  bool expect) {
  
  token_it begin = range.begin;
  result_t<node_ptr_t> result = binary_expression(range, expect);

  while(result) {
    range.advance(result);

    if(range.advance_if(tk_sym_ampamp)) {
      auto rhs = binary_expression(range, true);
      range.advance(rhs);

      auto binary = std::make_unique<node_binary_t>();
      binary->op = expr_op_log_and;
      binary->a = std::move(result->attr);
      binary->b = std::move(rhs->attr);
      result = make_result(begin, range.begin, std::move(binary));

    } else 
      break;
  }

  return result;
}

result_t<node_ptr_t> grammar_t::logical_or_expression(range_t range,
  bool expect) {

  token_it begin = range.begin;
  result_t<node_ptr_t> result = logical_and_expression(range, expect);
  while(result) {
    range.advance(result);

    if(range.advance_if(tk_sym_ampamp)) {
      auto rhs = logical_and_expression(range, true);
      range.advance(rhs);

      auto binary = std::make_unique<node_binary_t>();
      binary->op = expr_op_log_or;
      binary->a = std::move(result->attr);
      binary->b = std::move(rhs->attr);
      result = make_result(begin, range.begin, std::move(binary));

    } else 
      break;
  }

  return result;
}


////////////////////////////////////////////////////////////////////////////////

expr_op_t switch_assign(tk_kind_t kind) {
  expr_op_t op = expr_op_none;
  switch(kind) {
    case tk_sym_eq:        op = expr_op_assign;     break;
    case tk_sym_stareq:    op = expr_op_assign_mul; break;
    case tk_sym_slasheq:   op = expr_op_assign_div; break;
    case tk_sym_percenteq: op = expr_op_assign_mod; break;
    case tk_sym_pluseq:    op = expr_op_assign_add; break;
    case tk_sym_minuseq:   op = expr_op_assign_sub; break;
    case tk_sym_ltlteq:    op = expr_op_assign_shl; break;
    case tk_sym_gtgteq:    op = expr_op_assign_shr; break;
    case tk_sym_ampeq:     op = expr_op_assign_and; break;
    case tk_sym_pipeeq:    op = expr_op_assign_or;  break;
    case tk_sym_careteq:   op = expr_op_assign_xor; break;
    default:                                        break;
  }
  return op;
}

result_t<node_ptr_t> grammar_t::assignment_expression(range_t range, 
  bool expect) {

  token_it begin = range.begin;
  result_t<node_ptr_t> result;
  if(auto a = logical_or_expression(range, expect)) {
    range.advance(a);

    if(auto op = parse_switch(range, switch_assign)) {
      range.advance(op);

      // Match an initializer clause.
      auto b = initializer_clause(range, true);
      range.advance(b);

      auto assign = std::make_unique<node_assign_t>();
      assign->a = std::move(a->attr);
      assign->b = std::move(b->attr);

      a->attr = std::move(assign);

    } else if(range.advance_if(tk_sym_question)) {
      // Start of a ternary expression ? :
      auto b = assignment_expression(range, true);
      range.advance(b);

      if(!range.advance_if(tk_sym_col))
        throw_error(range.begin, "expected ':' in conditional-expression");

      auto c = assignment_expression(range, true);
      range.advance(c);

      auto ternary = std::make_unique<node_ternary_t>();
      ternary->a = std::move(a->attr);
      ternary->b = std::move(b->attr);
      ternary->c = std::move(c->attr);

      a->attr = std::move(ternary);
    }

    result = make_result(begin, range.begin, std::move(a->attr));
  }

  return result;
}

result_t<node_ptr_t> grammar_t::expression(range_t range, bool expect) {
  token_it begin = range.begin;
  result_t<node_ptr_t> result;

  if(auto expr = assignment_expression(range, expect)) {
    range.advance(expr);

    while(range.advance_if(tk_sym_comma)) {
      auto expr2 = assignment_expression(range, true);
      range.advance(expr2);

      auto binary = std::make_unique<node_binary_t>();
      binary->op = expr_op_sequence;
      binary->a = std::move(expr->attr);
      binary->b = std::move(expr2->attr);
      expr->attr = std::move(binary);
    }

    result = make_result(begin, range.begin, std::move(expr->attr));
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////

result_t<node_list_t> grammar_t::paren_initializer(range_t range) {
  token_it begin = range.begin;
  result_t<node_list_t> result;
  if(auto paren = parse_paren(range)) {
    range.advance(paren);

    auto list = init_list(paren->attr);
    result = make_result(begin, range.begin, std::move(list->attr));
  }
  return result;
}

result_t<node_ptr_t> grammar_t::paren_expression(range_t range) {
  token_it begin = range.begin;
  result_t<node_ptr_t> result;

  if(auto paren = parse_paren(range)) {
    range.advance(paren);
    range_t range2 = paren->attr;

    if(auto expr = expression(range2, true)) {
      range2.advance(expr);

      if(range2)
        unexpected_token(range2.begin, "expression");

      result = make_result(begin, range.begin, std::move(expr->attr));

    } else
      throw_error(range.begin, "expected expression");
  }
  
  return result;
}

result_t<node_ptr_t> grammar_t::braced_init_list(range_t range) {
  token_it begin = range.begin;
  result_t<node_ptr_t> result;

  if(auto brace = parse_brace(range)) {
    range.advance(brace);
    range_t range2 = brace->attr;

    // Support a braced initializer with a trailing , as long as there are
    // other tokens.
    if(range2.end - 1 > range2.begin && tk_sym_comma == range2.end[-1])
      --range2.end;

    node_list_t init_list;
  }
  return result;
}

result_t<node_ptr_t> grammar_t::initializer_clause(range_t range, bool expect) {
  result_t<node_ptr_t> result = braced_init_list(range);
  if(!result) result = assignment_expression(range, expect);
  return result;
}

result_t<node_list_t> grammar_t::init_list(range_t range) {
  // Must consume all elements.
  token_it begin = range.begin;
  node_list_t list;
  if(auto expr = initializer_clause(range, false)) {
    range.advance(expr);
    list.push_back(std::move(expr->attr));

    while(range.advance_if(tk_sym_comma)) {
      auto expr2 = initializer_clause(range, true);
      range.advance(expr2);
      list.push_back(std::move(expr2->attr));
    }
  }

  if(range)
    unexpected_token(range.begin, "initializer-list");

  return make_result(begin, range.begin, std::move(list));
}

////////////////////////////////////////////////////////////////////////////////

void grammar_t::throw_error(token_it pos, const char* msg) {
  printf("error thrown %s\n", msg);
  abort();
}

void grammar_t::unexpected_token(token_it pos, const char* msg) {
  printf("unexpected token in %s\n", msg);
  abort();

}

////////////////////////////////////////////////////////////////////////////////

parse_t parse_expression(const char* begin, const char* end) {
  // Tokenize the input.
  parse_t parse;
  parse.tokens = parse.tokenizer.tokenize({ begin, end });

  // Parse the tokens.
  grammar_t g;
  range_t range { 
    parse.tokens.data(), 
    parse.tokens.data() + parse.tokens.size() 
  };
  auto expr = g.expression(range, true);
  range.advance(expr);
  if(range)
    g.unexpected_token(range.begin, "expression");
  parse.root = std::move(expr->attr);

  return std::move(parse);
}


parse_t parse_expression(const char* str) {
  return parse_expression(str, str + strlen(str));
}

END_APEX_NAMESPACE

