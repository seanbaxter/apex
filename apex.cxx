#include <cstdio>
#include <algorithm>
#include "parse.hxx"

// Label each independent variable?

struct ad_t {
  enum kind_t {
    kind_tape,
    kind_literal,
    kind_unary,
    kind_binary,
    kind_sq,
    kind_func
  };
  kind_t kind;

  ad_t(kind_t kind) : kind(kind) { }
};
typedef std::unique_ptr<ad_t> ad_ptr_t;

struct ad_tape_t : ad_t {
  // Return a value from the tape with this index.
  ad_tape_t(int index) : ad_t(kind_tape), index(index) { }
  int index;
};

struct ad_literal_t : ad_t {
  // Yield a literal value.
  ad_literal_t(double x) : ad_t(kind_literal), x(x) { }
  double x;
};

struct ad_unary_t : ad_t {
  ad_unary_t(const char* op, ad_ptr_t a) :
    ad_t(kind_unary), op(op), a(std::move(a)) { }

  const char* op;
  ad_ptr_t a;
};

struct ad_binary_t : ad_t {
  ad_binary_t(const char* op, ad_ptr_t a, ad_ptr_t b) : 
    ad_t(kind_binary), op(op), a(std::move(a)), b(std::move(b)) { }

  const char* op;
  ad_ptr_t a, b;
};

struct ad_sq_t : ad_t {
  ad_sq_t(ad_ptr_t a) : ad_t(kind_sq), a(std::move(a)) { }
  ad_ptr_t a;
}; 

struct ad_func_t : ad_t {
  ad_func_t(std::string f, ad_ptr_t a, ad_ptr_t b = nullptr) : 
    ad_t(kind_func), f(std::move(f)), a(std::move(a)), b(std::move(b)) { }

  std::string f;
  ad_ptr_t a, b;
};

struct ad_builder_t {

  int literal_node(double x);

  // Operators
  int add(int a, int b);
  int sub(int a, int b);
  int mul(int a, int b);
  int div(int a, int b);
  int negate(int a);

  // Calls to elementary functions.
  int sq(int a);
  int sqrt(int a);
  int exp(int a);
  int ln(int a);
  int sin(int a);
  int cos(int a);
  int tan(int a);
  int sinh(int a);
  int cosh(int a);
  int tanh(int a);
  int pow(int a, int b);
  int norm(const int* p, int count);
  
  ad_ptr_t val(int index);
  ad_ptr_t literal(double x);
  ad_ptr_t add(ad_ptr_t a, ad_ptr_t b);
  ad_ptr_t sub(ad_ptr_t a, ad_ptr_t b);
  ad_ptr_t mul(ad_ptr_t a, ad_ptr_t b);
  ad_ptr_t div(ad_ptr_t a, ad_ptr_t b);
  ad_ptr_t rcp(ad_ptr_t a);
  ad_ptr_t sq(ad_ptr_t a);
  ad_ptr_t func(const char* name, ad_ptr_t a, ad_ptr_t b = nullptr);

  std::string str(const apex::node_t* node);

  int recurse(const apex::node_ident_t* node);
  int recurse(const apex::node_member_t* node);
  int recurse(const apex::node_subscript_t* node);
  int recurse(const apex::node_unary_t* node);
  int recurse(const apex::node_binary_t* node);
  int recurse(const apex::node_call_t* node);
  int recurse(const apex::node_t* node);

  void throw_error(const apex::node_t* node, const char* msg);

  struct item_t {
    // NOTE: Expression to compute value in terms of other slots and 
    // the independent variables.

    // The expression to execute to compute this dependent variable's value.
    // This is evaluated during the upsweep when creating the tape from the 
    // independent variables and moving through all subexpressions.
    ad_ptr_t val;

    // When updating the gradient of the parent, this tape item loops over each
    // of its dependent variables and performs a chain rule increment.
    // It calls grad(index, coef) on each index. This recurses, down to the
    // independent vars, multiplying in the coef at each recurse. 

    // When we hit an independent var, the grads array is empty (although it
    // may be empty otherwise) and we simply perform += coef into the slot
    // corresponding to the independent variable in the gradient array.
    struct grad_t {
      int index;
      ad_ptr_t coef;
    };
    std::vector<grad_t> grads;
  };
  int push_item(item_t item) {
    int count = tape.size();
    tape.push_back(std::move(item));
    return count;
  }

  int find_var(const apex::node_t* node, std::string name);

  // Text of the AD formula.
  std::string text;

  // Each of the independent variables in gradient order.
  std::vector<std::string> var_names;

  // The first var_names.size() items encode independent variables.
  std::vector<item_t> tape;
};

int ad_builder_t::literal_node(double x) {
  item_t item { };
  item.val = literal(x);
  return push_item(std::move(item));
}

int ad_builder_t::add(int a, int b) {
  item_t item { };
  item.val = add(val(a), val(b));
  item.grads.push_back({
    a,
    literal(1)
  });
  item.grads.push_back({
    b,
    literal(1)
  });
  return push_item(std::move(item));
}

int ad_builder_t::sub(int a, int b) {
  item_t item { };
  item.val = sub(val(a), val(b));
  item.grads.push_back({
    a,
    literal(1)
  });
  item.grads.push_back({
    b,
    literal(-1)
  });
  return push_item(std::move(item));
}

int ad_builder_t::mul(int a, int b) {
  // grad (a * b) = a grad b + b grad a.
  item_t item { };
  item.val = mul(val(a), val(b));
  item.grads.push_back({
    b,      // a * grad b
    val(a)
  });
  item.grads.push_back({
    a,      // b * grad a
    val(b)
  });
  return push_item(std::move(item));
}

int ad_builder_t::div(int a, int b) {
  // grad (a / b) = 1 / b * grad a - a / b^2 * grad b.
  item_t item { };
  item.val = div(val(a), val(b));
  item.grads.push_back({
    // 1 / b * grad a.
    a,
    rcp(val(b)) 
  });
  item.grads.push_back({
    // a / b^2 * grad b.
    b,
    div(val(a), sq(val(b)))
  });
  return push_item(std::move(item));
}

int ad_builder_t::negate(int a) {
  item_t item { };
  item.val = mul(literal(-1), val(a));
  item.grads.push_back({
    a,
    literal(-1)
  });
  return push_item(std::move(item));
}

////////////////////////////////////////////////////////////////////////////////
// Elementary functions

int ad_builder_t::sq(int a) {
  item_t item { };
  item.val = sq(val(a));
  item.grads.push_back({
    // grad (a^2) = 2 * a grad a
    a,
    mul(literal(2), val(a))
  });
  return push_item(std::move(item));
}

int ad_builder_t::sqrt(int a) {
  item_t item { };
  item.val = func("std::sqrt", val(a));
  item.grads.push_back({
    // .5 / sqrt(a) * grad a
    a,
    div(literal(.5), func("std::sqrt", val(a)))
  });
  return push_item(std::move(item));
}

int ad_builder_t::exp(int a) {
  item_t item { };
  item.val = func("std::exp", val(a));
  item.grads.push_back({
    // exp(a) * grad a
    a,
    func("std::exp", val(a))
  });
  return push_item(std::move(item));
}

int ad_builder_t::ln(int a) {
  // grad (ln a) = grad a / a
  item_t item { };
  item.val = func("std::ln", val(a));
  item.grads.push_back({
    a,
    rcp(val(a))
  });
  return push_item(std::move(item));
}

int ad_builder_t::sin(int a) {
  item_t item { };
  item.val = func("std::sin", val(a));
  item.grads.push_back({
    a,
    func("std::cos", val(a))
  });
  return push_item(std::move(item));
}

int ad_builder_t::cos(int a) {
  item_t item { };
  item.val = func("std::cos", val(a));
  item.grads.push_back({
    a,
    mul(literal(-1), func("std::sin", val(a)))
  });
  return push_item(std::move(item));
}

int ad_builder_t::tan(int a) {
  item_t item { };
  item.val = func("std::tan", val(a));
  item.grads.push_back({
    a,
    sq(rcp(func("std::cos", val(a))))
  });
  return push_item(std::move(item));
}

int ad_builder_t::sinh(int a) {
  item_t item { };
  item.val = func("std::sinh", val(a));
  item.grads.push_back({
    a,
    func("std::cosh", val(a))
  });
  return push_item(std::move(item));
}

int ad_builder_t::cosh(int a) {
  item_t item { };
  item.val = func("std::cosh", val(a));
  item.grads.push_back({
    a,
    func("std::sinh", val(a))
  });
  return push_item(std::move(item));
}

int ad_builder_t::tanh(int a) {
  item_t item { };
  item.val = func("std::tanh", val(a));
  item.grads.push_back({
    a,
    sub(literal(1), sq(func("std::tanh", val(a))))
  });
  return push_item(std::move(item));
}

int ad_builder_t::pow(int a, int b) {
  item_t item { };
  item.val = func("std::pow", val(a), val(b));
  item.grads.push_back({
    // d/dx (a**b) = b a**(b - 1) da/dx
    a,
    mul(val(b), func("std::pow", val(a), sub(val(b), literal(1))))
  });
  item.grads.push_back({
    // d/dx (a**b) = a**b ln a db/dx
    b,
    mul(func("std::pow", val(a), val(b)), func("std::ln", val(a)))
  });
  return push_item(std::move(item));
}

int ad_builder_t::norm(const int* p, int count) {
  item_t item { };

  // Square and accumulate each argument.
  ad_ptr_t x = sq(val(p[0]));
  for(int i = 1; i < count; ++i)
    x = add(std::move(x), sq(val(p[i])));

  // Take its sqrt.
  item.val = func("std::sqrt", std::move(x));

  // Differentiate with respect to each argument.
  // The derivative is f_i * grad f_i / norm(f).
  // We compute the norm in this tape item during the upsweep, so load it.
  // We have a 1 / norm common subexpression--this can be eliminated by the
  // optimizer, but may be added to the tape as its own value.
  int index = tape.size();
  for(int i = 0; i < count; ++i) {
    item.grads.push_back({
      p[i],
      div(val(p[i]), val(index))
    });
  }
  return push_item(std::move(item));
}

std::string ad_builder_t::str(const apex::node_t* node) {
  switch(node->kind) {
    case apex::node_t::kind_ident:
      return static_cast<const apex::node_ident_t*>(node)->s;

    case apex::node_t::kind_member: {
      const auto* member = static_cast<const apex::node_member_t*>(node);
      return str(member->lhs.get()) + "." + member->member;
    }

    case apex::node_t::kind_subscript: {
      const auto* subscript = static_cast<const apex::node_subscript_t*>(node);
      if(1 != subscript->args.size())
        throw_error(node, "subscript must have 1 index");
      return str(subscript->lhs.get()) + 
        "[" + str(subscript->args[0].get()) + "]";
    }

    case apex::node_t::kind_int: {
      const auto* int_ = static_cast<const apex::node_int_t*>(node);
      return std::to_string(int_->ui);
    }

    default:
      throw_error(node, "unsupported identifier kind");
  }
}

int ad_builder_t::recurse(const apex::node_unary_t* node) {
  int a = recurse(node->a.get());
  int c = -1;
  switch(node->op) {
    case apex::expr_op_negate:
      c = negate(a);
      break;

    default:
      throw_error(node, "unsupported operator XXX");
  } 
  return c;
}

int ad_builder_t::recurse(const apex::node_binary_t* node) {
  int a = recurse(node->a.get());
  int b = recurse(node->b.get());
  int c = -1;

  switch(node->op) {
    case apex::expr_op_add:
      c = add(a, b);
      break;

    case apex::expr_op_sub:
      c = sub(a, b);
      break;

    case apex::expr_op_mul:
      c = mul(a, b);
      break;

    case apex::expr_op_div:
      c = div(a, b);
      break;

    default:
      throw_error(node, "unsupported operator XXX");
  }
  return c;
}

int ad_builder_t::recurse(const apex::node_call_t* node) {
  std::string func_name = str(node->f.get());
  std::vector<int> args(node->args.size());
  for(int i = 0; i < node->args.size(); ++i)
    args[i] = recurse(node->args[i].get());

  #define GEN_CALL_1(s) \
    if(#s == func_name) { \
      if(1 != args.size()) \
        throw_error(node, #s "() requires 1 argument"); \
      return s(args[0]); \
    }

  GEN_CALL_1(sq)
  GEN_CALL_1(sqrt)
  GEN_CALL_1(exp)
  GEN_CALL_1(ln)
  GEN_CALL_1(sin)
  GEN_CALL_1(cos)
  GEN_CALL_1(tan)
  GEN_CALL_1(sinh)
  GEN_CALL_1(cosh)
  GEN_CALL_1(tanh)
 
  #undef GEN_CALL_1

  if("pow" == func_name) {
    if(2 != node->args.size())
      throw_error(node, "pow() requires 2 arguments");
    return pow(args[0], args[1]);

  } else if("norm" == func_name) {
    // Allow 1 or more arguments.
    if(!node->args.size())
      throw_error(node, "norm() requires 1 or more arguments");
    return norm(args.data(), args.size());

  } else {
    throw_error(node, "unknown function XXX");
  }
}

int ad_builder_t::recurse(const apex::node_t* node) {
  int result = -1;
  switch(node->kind) {
    case apex::node_t::kind_float:
      result = literal_node(static_cast<const apex::node_float_t*>(node)->ld);
      break;

    case apex::node_t::kind_int:
      result = literal_node(static_cast<const apex::node_int_t*>(node)->ui);
      break;      

    case apex::node_t::kind_ident:
    case apex::node_t::kind_member:
    case apex::node_t::kind_subscript:
      // Don't add a new tape item for independent variables--these get 
      // provisioned in order at the start.
      result = find_var(node, str(node));
      break;

    case apex::node_t::kind_unary:
      result = recurse(static_cast<const apex::node_unary_t*>(node));
      break;

    case apex::node_t::kind_binary:
      result = recurse(static_cast<const apex::node_binary_t*>(node));
      break;

    case apex::node_t::kind_call:
      result = recurse(static_cast<const apex::node_call_t*>(node));
      break;

    default:
      break;
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////

ad_ptr_t ad_builder_t::val(int index) {
  // Return a value from the tape.
  return std::make_unique<ad_tape_t>(index);
}

ad_ptr_t ad_builder_t::literal(double x) {
  return std::make_unique<ad_literal_t>(x);
}

ad_ptr_t ad_builder_t::add(ad_ptr_t a, ad_ptr_t b) {
  return std::make_unique<ad_binary_t>("+", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::sub(ad_ptr_t a, ad_ptr_t b) {
  return std::make_unique<ad_binary_t>("-", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::mul(ad_ptr_t a, ad_ptr_t b) {
  return std::make_unique<ad_binary_t>("*", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::div(ad_ptr_t a, ad_ptr_t b) {
  return std::make_unique<ad_binary_t>("/", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::rcp(ad_ptr_t a) {
  return div(literal(1), std::move(a));
}

ad_ptr_t ad_builder_t::sq(ad_ptr_t a) {
  return std::make_unique<ad_sq_t>(std::move(a));
}

ad_ptr_t ad_builder_t::func(const char* f, ad_ptr_t a, ad_ptr_t b) {
  return std::make_unique<ad_func_t>(f, std::move(a), std::move(b));
}

////////////////////////////////////////////////////////////////////////////////

void ad_builder_t::throw_error(const apex::node_t* node, const char* msg) {


}

int ad_builder_t::find_var(const apex::node_t* node, std::string name) {
  auto it = std::find(var_names.begin(), var_names.end(), name);
  if(var_names.end() == it)
    throw_error(node, "unknown variable name");
  return it - var_names.begin();
}

int main() {
  std::string s = "x + 3 * z * x / sin(y + z)";
  auto p = apex::parse_expression(s.c_str());

  ad_builder_t ad_builder;
  ad_builder.text = s;
  ad_builder.var_names.push_back("x");
  ad_builder.var_names.push_back("y");
  ad_builder.var_names.push_back("z");
  ad_builder.tape.resize(3);

  ad_builder.recurse(p.root.get());

  return 0;
}
