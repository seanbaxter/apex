#include <apex/util.hxx>
#include <apex/parse.hxx>

BEGIN_APEX_NAMESPACE

struct ad_exeption_t : std::runtime_error {
  ad_exeption_t(const std::string& err) : std::runtime_error(err) { }
};

struct ad_t {
  enum kind_t {
    kind_tape,
    kind_component,
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
  ad_tape_t(int index) : ad_t(kind_tape), index(index) { }
  static bool classof(const ad_t* ad) { return kind_tape == ad->kind; }

  int index;
};

struct ad_component_t : ad_t {
  ad_component_t(int index) : ad_t(kind_component), index(index) { }
  static bool classof(const ad_t* ad) { return kind_component == ad->kind; }

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

// Each primary input may be a scalar (dim 0) or a vector (dim > 0).
// autodiff_codegen.hxx uses introspection to parse these out of 
// the argument type.
struct autodiff_var_t {
  std::string name;
  int dim;
};

struct autodiff_t {
  struct item_t {
    // The dimension of the tape item. 
    // 0 == dim for scalar. dim > 0 for vector.
    int dim;

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

  // The first var_names.size() items encode independent variables.
  std::vector<autodiff_var_t> vars;
  std::vector<item_t> tape;
};

autodiff_t make_autodiff(const std::string& formula, 
  const std::vector<autodiff_var_t>& vars);

std::string print_ad(const ad_t* ad, int indent = 0);
std::string print_autodiff(const autodiff_t& autodiff);

END_APEX_NAMESPACE
