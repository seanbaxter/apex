#include <apex/util.hxx>
#include <apex/parse.hxx>

BEGIN_APEX_NAMESPACE

struct ad_exeption_t : std::runtime_error {
  ad_exeption_t(const std::string& err) : std::runtime_error(err) { }
};

struct ad_t {
  enum kind_t {
    kind_tape,
    kind_literal,
    kind_unary,
    kind_binary,
    kind_func
  };
  kind_t kind;
  
  ad_t(kind_t kind) : kind(kind) { }

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
typedef std::unique_ptr<ad_t> ad_ptr_t;

struct ad_tape_t : ad_t {
  // Return a value from the tape with this index.
  ad_tape_t(int index) : ad_t(kind_tape), index(index) { }
  static bool classof(const ad_t* ad) { return kind_tape == ad->kind; }

  int index;
};

struct ad_literal_t : ad_t {
  ad_literal_t(double x) : ad_t(kind_literal), x(x) { }
  static bool classof(const ad_t* ad) { return kind_literal == ad->kind; }

  double x;
};

struct ad_unary_t : ad_t {
  ad_unary_t(const char* op, ad_ptr_t a) :
    ad_t(kind_unary), op(op), a(std::move(a)) { }
  static bool classof(const ad_t* ad) { return kind_unary == ad->kind; }

  const char* op;
  ad_ptr_t a;
};

struct ad_binary_t : ad_t {
  ad_binary_t(const char* op, ad_ptr_t a, ad_ptr_t b) : 
    ad_t(kind_binary), op(op), a(std::move(a)), b(std::move(b)) { }
  static bool classof(const ad_t* ad) { return kind_binary == ad->kind; }

  const char* op;
  ad_ptr_t a, b;
};

struct ad_func_t : ad_t {
  ad_func_t(std::string f) : ad_t(kind_func), f(std::move(f)) { }
  static bool classof(const ad_t* ad) { return kind_func == ad->kind; }

  std::string f;
  std::vector<ad_ptr_t> args;
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

  std::string str(const parse::node_t* node);

  int recurse(const parse::node_ident_t* node);
  int recurse(const parse::node_member_t* node);
  int recurse(const parse::node_subscript_t* node);
  int recurse(const parse::node_unary_t* node);
  int recurse(const parse::node_binary_t* node);
  int recurse(const parse::node_call_t* node);
  int recurse(const parse::node_t* node);

  void process(const std::string& formula, std::vector<std::string> var_names);
  void process(const parse::node_t* node, std::vector<std::string> var_names);

  void throw_error(const parse::node_t* node, const char* msg);

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

  int find_var(const parse::node_t* node, std::string name);

  // Text of the AD formula.
  std::string text;

  // Each of the independent variables in gradient order.
  std::vector<std::string> var_names;

  // The first var_names.size() items encode independent variables.
  std::vector<item_t> tape;
};

std::string print_ad(const ad_t* ad, int indent = 0);
std::string print_ad(const ad_builder_t& ad_builder);

END_APEX_NAMESPACE
