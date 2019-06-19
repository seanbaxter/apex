#include <apex/autodiff.hxx>
#include <sstream>
#include <cstdarg>
#include <map>
#include <algorithm>

BEGIN_APEX_NAMESPACE

using namespace parse;


struct ad_builder_t : autodiff_t {
  typedef autodiff_t::item_t item_t;
  typedef autodiff_t::item_t::grad_t grad_t;
  
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
  int log(int a);
  int sin(int a);
  int cos(int a);
  int tan(int a);
  int sinh(int a);
  int cosh(int a);
  int tanh(int a);
  int abs(int a);
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

  std::string str(const parse::node_t* node);

  int recurse(const parse::node_ident_t* node);
  int recurse(const parse::node_member_t* node);
  int recurse(const parse::node_subscript_t* node);
  int recurse(const parse::node_unary_t* node);
  int recurse(const parse::node_binary_t* node);
  int recurse(const parse::node_call_t* node);
  int recurse(const parse::node_t* node);

  void throw_error(const parse::node_t* node, const char* fmt, ...);

  int push_item(item_t item) {
    int count = tape.size();
    tape.push_back(std::move(item));
    return count;
  }

  int find_var(const parse::node_t* node, std::string name);

  // If the tokenizer is provided we can print error messages that are
  // line/col specific.
  const tok::tokenizer_t* tokenizer = nullptr;

  // Store each literal value once. This doesn't effect the computation 
  // directly, but is helpful for subexpression elimination.
  std::map<double, int> literal_map;

  enum op_name_t {
    op_name_tape,
    op_name_literal,
    op_name_add,
    op_name_sub,
    op_name_mul,
    op_name_div,
    op_name_negate,
    op_name_sq,
    op_name_sqrt,
    op_name_exp,
    op_name_log,
    op_name_sin,
    op_name_cos,
    op_name_tan,
    op_name_sinh,
    op_name_cosh,
    op_name_tanh,
    op_name_abs,
    op_name_pow,
  };

  union op_key_t {
    struct {
      op_name_t name : 8;
      uint a         : 28;
      uint b         : 28; 
    };
    uint64_t bits;
  };

  std::optional<int> find_cse(op_name_t op_name, int a, int b = -1);
  std::optional<int> find_literal(double x);

  // Map each operation to the location in the tape where its value is stored.
  // We only build this structure during the upsweep when computing the tape
  // values. We won't necessarily match common subexpressions in partial 
  // derivatives, because we don't want to memoize all those fragments as it
  // will consume more storage than we're prepared to give.
  std::map<uint64_t, int> cse_map;
};


////////////////////////////////////////////////////////////////////////////////
// TODO: Register each tape insertion with the common subexpression elimination
// (CSE) map, so that find_cse will work.

int ad_builder_t::literal_node(double x) {
  item_t item { };
  item.val = literal(x);
  return push_item(std::move(item));
}

int ad_builder_t::add(int a, int b) {
  if(auto cse = find_cse(op_name_add, a, b))
    return *cse;

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
  // Nip this in the bud.
  if(a == b)
    return literal_node(0);
  
  if(auto cse = find_cse(op_name_sub, a, b))
    return *cse;

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
  if(auto cse = find_cse(op_name_mul, a, b))
    return *cse;

  // The sq operator is memoized, so prefer that.
  if(a == b)
    return sq(a);

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
  if(auto cse = find_cse(op_name_div, a, b))
    return *cse;

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

int ad_builder_t::log(int a) {
  // grad (ln a) = grad a / a
  item_t item { };
  item.val = func("std::log", val(a));
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

int ad_builder_t::abs(int a) {
  item_t item { };
  item.val = func("std::abs", val(a));
  item.grads.push_back({
    a,    // d/dx abs(x) = x / abs(x)
    div(val(a), func("std::abs", val(a)))
  });
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
    mul(func("std::pow", val(a), val(b)), func("std::log", val(a)))
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

std::string ad_builder_t::str(const node_t* node) {
  switch(node->kind) {
    case node_t::kind_ident:
      return static_cast<const node_ident_t*>(node)->s;

    case node_t::kind_member: {
      const auto* member = static_cast<const node_member_t*>(node);
      return str(member->lhs.get()) + "." + member->member;
    }

    case node_t::kind_subscript: {
      const auto* subscript = static_cast<const node_subscript_t*>(node);
      if(1 != subscript->args.size())
        throw_error(node, "subscript must have 1 index");
      return str(subscript->lhs.get()) + 
        "[" + str(subscript->args[0].get()) + "]";
    }

    case node_t::kind_number: {
      const auto* number = static_cast<const node_number_t*>(node);
      return number->x.to_string();
    }

    default:
      throw_error(node, "unsupported identifier kind");
  }
}

int ad_builder_t::recurse(const node_unary_t* node) {
  int a = recurse(node->a.get());
  int c = -1;
  switch(node->op) {
    case expr_op_negate:
      c = negate(a);
      break;

    default:
      throw_error(node, "unsupported unary %s", expr_op_names[node->op]);
  } 
  return c;
}

int ad_builder_t::recurse(const node_binary_t* node) {
  int a = recurse(node->a.get());
  int b = recurse(node->b.get());
  int c = -1;

  switch(node->op) {
    case expr_op_add:
      c = add(a, b);
      break;

    case expr_op_sub:
      c = sub(a, b);
      break;

    case expr_op_mul:
      c = mul(a, b);
      break;

    case expr_op_div:
      c = div(a, b);
      break;

    default:
      throw_error(node, "unsupported binary %s", expr_op_names[node->op]);
  }
  return c;
}

int ad_builder_t::recurse(const node_call_t* node) {
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
  GEN_CALL_1(log)
  GEN_CALL_1(sin)
  GEN_CALL_1(cos)
  GEN_CALL_1(tan)
  GEN_CALL_1(sinh)
  GEN_CALL_1(cosh)
  GEN_CALL_1(tanh)
  GEN_CALL_1(abs)
 
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
    throw_error(node, "unknown function '%s'", func_name.c_str());
  }
}

int ad_builder_t::recurse(const node_t* node) {
  int result = -1;
  switch(node->kind) {
    case node_t::kind_number: {
      auto* number = node->as<node_number_t>();
      result = literal_node(number->x.convert<double>());
      break;
    } 

    case node_t::kind_ident:
    case node_t::kind_member:
    case node_t::kind_subscript:
      // Don't add a new tape item for independent variables--these get 
      // provisioned in order at the start.
      result = find_var(node, str(node));
      break;

    case node_t::kind_unary:
      result = recurse(static_cast<const node_unary_t*>(node));
      break;

    case node_t::kind_binary:
      result = recurse(static_cast<const node_binary_t*>(node));
      break;

    case node_t::kind_call:
      result = recurse(static_cast<const node_call_t*>(node));
      break;

    default:
      break;
  }
  return result;
}

autodiff_t make_autodiff(const parse_t& parse, 
  const std::vector<std::string>& var_names) {

  ad_builder_t ad_builder;
  ad_builder.tokenizer = &parse.tokenizer;
  ad_builder.var_names = var_names;
  ad_builder.tape.resize(ad_builder.var_names.size());
  ad_builder.recurse(parse.root.get());

  return std::move(ad_builder);
}

autodiff_t make_autodiff(const std::string& formula,
  const std::vector<std::string>& var_names) {

  auto p = parse::parse_expression(formula.c_str()); 
  return make_autodiff(p, std::move(var_names));
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
  auto* a2 = a->as<ad_literal_t>();
  auto* b2 = b->as<ad_literal_t>();
  if(a2 && b2)
    return literal(a2->x + b2->x);
  else 
    return std::make_unique<ad_binary_t>("+", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::sub(ad_ptr_t a, ad_ptr_t b) {
  auto* a2 = a->as<ad_literal_t>();
  auto* b2 = b->as<ad_literal_t>();
  if(a2 && b2)
    return literal(a2->x - b2->x);
  return std::make_unique<ad_binary_t>("-", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::mul(ad_ptr_t a, ad_ptr_t b) {
  auto* a2 = a->as<ad_literal_t>();
  auto* b2 = b->as<ad_literal_t>();
  if(a2 && b2)
    return literal(a2->x * b2->x);
  return std::make_unique<ad_binary_t>("*", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::div(ad_ptr_t a, ad_ptr_t b) {
  auto* a2 = a->as<ad_literal_t>();
  auto* b2 = b->as<ad_literal_t>();
  if(a2 && b2)
    return literal(a2->x / b2->x);
  return std::make_unique<ad_binary_t>("/", std::move(a), std::move(b));
}

ad_ptr_t ad_builder_t::rcp(ad_ptr_t a) {
  if(auto* a2 = a->as<ad_literal_t>())
    return literal(1 / a2->x);
  else
    return div(literal(1), std::move(a));
}

ad_ptr_t ad_builder_t::sq(ad_ptr_t a) {
  if(auto* a2 = a->as<ad_literal_t>())
    return literal(a2->x * a2->x);
  else
    return func("apex::sq", std::move(a));
}

ad_ptr_t ad_builder_t::func(const char* f, ad_ptr_t a, ad_ptr_t b) {
  // TODO: Perform constant folding?

  auto node = std::make_unique<ad_func_t>(f);
  node->args.push_back(std::move(a));
  if(b) node->args.push_back(std::move(b));
  return node;
}

////////////////////////////////////////////////////////////////////////////////

void ad_builder_t::throw_error(const node_t* node, const char* fmt, ...) {
  // Get the user's error message.
  va_list args;
  va_start(args, fmt);
  std::string msg = vformat(fmt, args);
  va_end(args);

  // If the tokenizer is available, print a location message.
  if(tokenizer) {
    std::pair<int, int> linecol = tokenizer->token_linecol(node->loc);
    msg = format(
      "autodiff formula \"%s\"\n"
      "line %d col %d\n"
      "%s", 
      tokenizer->text.c_str(),
      linecol.first + 1,
      linecol.second + 1,
      msg.c_str()
    );
  }

  throw ad_exeption_t(msg);
}

int ad_builder_t::find_var(const node_t* node, std::string name) {
  auto it = std::find(var_names.begin(), var_names.end(), name);
  if(var_names.end() == it)
    throw_error(node, "unknown variable '%s'", name.c_str());
  return it - var_names.begin();
}

std::optional<int> ad_builder_t::find_cse(op_name_t op_name, int a, int b) {
  switch(op_name) {
    case op_name_add:
    case op_name_mul:
      // For these commutative operators, put the lower index on the left.
      // This improves CSE performance.
      if(a > b)
        std::swap(a, b);
      break;

    default:
      break;
  }

  op_key_t op_key { op_name, (uint)a, (uint)b };
  auto it = cse_map.find(op_key.bits);
  std::optional<int> index;
  if(cse_map.end() != it) {
    index = it->second;
  }
  return index;
}

std::optional<int> ad_builder_t::find_literal(double x) {
  auto it = literal_map.find(x);
  std::optional<int> index;
  if(literal_map.end() != it) {
    index = it->second;
  }
  return index;
}

////////////////////////////////////////////////////////////////////////////////

void print_ad(const ad_t* ad, std::ostringstream& oss, int indent) {
  for(int i = 0; i < indent; ++i)
    oss.write("  ", 2);

  if(auto* tape = ad->as<ad_tape_t>()) {
    oss<< "tape "<< tape->index<< "\n";

  } else if(auto* literal = ad->as<ad_literal_t>()) {
    oss<< "literal "<< literal->x<< "\n";

  } else if(auto* unary = ad->as<ad_unary_t>()) {
    oss<< "unary "<< unary->op<< "\n";
    print_ad(unary->a.get(), oss, indent + 1);

  } else if(auto* binary = ad->as<ad_binary_t>()) {
    oss<< "binary "<< binary->op<< "\n";
    print_ad(binary->a.get(), oss, indent + 1);
    print_ad(binary->b.get(), oss, indent + 1);

  } else if(auto* func = ad->as<ad_func_t>()) {
    oss<< func->f<< "()\n";
    for(const auto& arg : func->args)
      print_ad(arg.get(), oss, indent + 1);
  }
}

std::string print_ad(const ad_t* ad, int indent) {
  std::ostringstream oss;
  print_ad(ad, oss, indent);
  return oss.str();
}

std::string print_autodiff(const autodiff_t& autodiff) {
  // Print all non-terminal tape items.
  std::ostringstream oss;

  for(int i = autodiff.var_names.size(); i < autodiff.tape.size(); ++i) {
    const auto& item = autodiff.tape[i];

    oss<< "tape "<< i<< ":\n";

    // Print the value.
    oss<< "  value =\n";
    oss<< print_ad(item.val.get(), 2);

    // Print each gradient.
    for(const auto& grad : item.grads) {
      oss<< "  grad "<< grad.index<< " = \n";
      oss<< print_ad(grad.coef.get(), 2);
    }
  }

  return oss.str();
}

END_APEX_NAMESPACE
