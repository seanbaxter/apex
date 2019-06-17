# Reverse-mode automatic differentiation with Circle and Apex

```cpp
std::array<double, 2> my_grad(double x, double y) {
  return apex::autodiff_grad("sq(x / y) + sin(x)", { "x", "y" });
}
```

This example shows how to leverage the Apex DSL library to implement reverse-mode automatic differentation. The expression to differentiate is passed in as a compile-time string, and the primary inputs are provided as a vector of strings.

This illustrates shared object library development. There are three things you can count on:
1. No template metaprogramming.
1. No operator overloading.
1. No performance compromise.

Do all your development in an ordinary C++/Circle shared object project. Call into this shared object library during source translation and capture the returned IR. Lower the IR to code using Circle macros. This is a new way forward for DSLs in C++.

## Expression templates

In Standard C++, code must either be included in textual form or compiled to binary and linked with the program. Only the former form is generic--template libraries may be specialized to solve an application-specific problem.

EDSLs have been attempted by the C++ community for almost twenty years. In Standard C++, the [expression template](https://en.wikipedia.org/wiki/Expression_templates) idiom is used to repurpose the C++ syntax as a domain-specific language. Operator overloading and template metaprogramming are combined to capture the result of a subexpression as a _type_. For example, if either terminal in the subexpression `a + b` is an EDSL type, the result object of the addition expression is a type that includes details of the operator and of each operand. For example, `op_add_t<left_type, right_type>`, where the template arguments are EDSL operator types that recursively specify their operand types. The type of the result object for the full expression contains the information of a parse tree over that same expression input. The expression template may be traversed (as if one were traversing a parse tree) and some calculation performed at each node.

Expression templates are extremely difficult to write, error messages are opaque (mostly due to the hierarchical nature of the involved types) and) and build times are long. Most critically, expression-template EDSLs don't allow very complex compile-time transformations on the parse tree content. Once the expression template is built, the user remains limited by C++'s lack of compile-time imperative programming support. The user cannot lower the expression template to a rational IR, or build tree data structures, or run the content through optimizers or analyzers.

## The Apex vision for libraries

Circle's integrated interpreter and code reflection mechanisms establish a larger design space for libraries. _What is a library with Circle?_ **Any code that provides a service**.

As demonstrated with the [Tensor Compiler example](https://github.com/seanbaxter/circle/blob/master/gems/taco.md), a Circle program can dynamically link to a shared object library _at compile time_, use that library to solve an intricate problem (tensor contraction scheduling), then lower the resultin solution (the intermediate representation) to code using Circle's macro system.

Apex is a collection of services to help programmers develop this new form of library. Currently it includes a tokenizer and parser for a C++ subset (called the Apex grammar), as well as a reverse-mode automatic differentation package that serves as both an example for building new libraries and an ingredient for additional numerical computing development.

Functionality built using Apex presents itself to the application developer as an _embedded domain-specific language_. But the design of Apex EDSLs is very different from the design of expression templates: there is no operator overloading; there is no template metaprogramming; we don't try to co-opt C++ tokens into carrying DSL information.

The client communicates with the library by way of compile-time strings. The contents may be provided as literals or assembled from code and data using Circle's usual compile-time mechanisms. The library transforms the input text into code:

1. The Apex tokenizer lexes the text into tokens. The tokenizer code is in the shared object library `libapex.so`.
1. The Apex parser processes the tokens into a parse tree. The parser code is also in `libapex.so`. The parse tree is an light-weight class hierarchy. There are node types for unary and binary operators, function calls, terminals, and so on. It is not a template library.  
    Parse errors are formed by the parser--you don't get C++ frontend errors when the input is malformed, but Apex backend errors, which are cleaner and more relevant to the problem of parsing.
1. The EDSL library traverses the parse tree and performs semantic analysis. This is where the library intelligence resides. All usual programming tools are available to the library. It can interact with the file system, host interpreters to execute scripts, and so on. The library intelligence should be compiled into a shared object; Apex's autodiff package is compiled into `libapex.so`.  
    The output of the intelligence layer is the problem solution in an intermediate representation. This IR may be expressed using any C++ data structure. Because the library's shared object is loaded into the compiler's process during source translation, objects created by the intelligence layer occupy the same address space as the Circle interpreter, allowing those objects to be consumed and modified by meta code.
1. A codegen header supplies the interface between the client, the Apex tokenizer and parser, and the library intelligence. This header provides Circle macros to lower the EDSL operation from IR to expression code. 

Although this seems like an involved four-step pipeline, the first two components are reusable and provided by libraries. Even if you choose a different tokenizer or parser, you can use them from libraries. The intelligence layer establishes a nice separation of concerns, as you can develop it independently of the code generator. Finally, the codegen layer is very small, as all difficult work was pushed down and handled by shared object libraries.

A strength of this approach is that it requires very little Circle code, only a thin layer of macros for code generation. All the intelligence can be written with Standard C++ for portability and to ease migration into this new language.

## Autodiff for Circle

```cpp
#include <apex/autodiff_codegen.hxx>

std::array<double, 2> my_grad(double x, double y) {
  return apex::autodiff_grad("sq(x / y) * sin(x)", { "x", "y" });
}

int main() {
  auto grad = my_grad(.3, .5);
  printf("%f %f\n", grad[0], grad[1]);
  return 0;
}
```
```
$ circle grad1.cxx -I ../include -M ../Debug/libapex.so 
$ ./grad1
1.053170 0.425549
```

To use Apex's autodiff, pass the formula to differentiate as a string, followed by an `std::vector` of independent variable names. The result object is the total derivative gradient stored in an `std::array`. The names of the independent variables in the string must match their names in the scope of the call--don't let the string literal fool you, the EDSL is real code (although not C++ code), and the values of the independent variables are gathered from their expression names in the formula string.

After just two days of programming, this package supports these expressions and elementary functions:
* Binary + - * and /.
* Unary -.
* sq, sqrt, exp, log, sin, cos, tan, sinh, cosh, tanh, pow and norm functions.

The call to `autodiff_grad` has distinct compile-time and runtime phases. At compile time, the formula is tokenized and parsed; the parse tree is lowered by `ad_builder_t` to an IR called a "tape," and that tape is lowered by `autodiff_codegen.hxx` to code using Circle macros. At runtime, the independent variables are evaluated and the tape-generated code is executed, yielding the gradient. All scheduling is performed at compile time, and there is no runtime dependency on any part of the `libapex.so` library.

Reverse-mode differentation is essentially a sparse matrix problem. Each dependent variable/subexpression is a row in the sparse matrix (an item in the tape) with a non-zero column for each partial derivative we'll compute to complete the chain rule. When considered as a DAG traversal, the chain rule calculation involves propagating partials from the root down each edge, and incrementing a component of the gradient vector by the concatenated chain rule coefficient. When viewed as linear algebra, the entire gradient pass is a sparse back-propagation operation. 

The Apex autodiff example adopts the DAG view of the problem. The implementation performs best when the size of the DAG is small enough so that the gains of explicit scheduling of each individual back-propagation term more than offset the parallelism left on the table by not using an optimized sparse matrix back-propagation code.

However, the separation of autodiff intelligence and code generation permits selection of a back-propagation treatment most suitable for the particular primary inputs and expression graph. Calls into the autodiff library with different expressions may generate implementations utilizing different strategies, without the vexations of template metaprogramming.

## The autodiff code generator

[**autodiff_codegen.hxx**](../include/apex/autodigg_codegen.hxx)
```cpp
@macro auto autodiff_grad(std::string __fmt, 
  std::vector<std::string> __var_names) {

  // Construct the autodiff builder.
  @meta apex::ad_builder_t __ad_builder;
  @meta __ad_builder.process(__fmt, __var_names);

  // Generate and call into a metafunction. The meta argument is the
  // autodiff builder. The real arguments are the values of each of the
  // independent variables, evaluated in the scope of solve_autodiff's 
  // caller and supplied to autodiff_eval using parameter pack expansion. 
  return autodiff_eval(
    __ad_builder, 
    @expression(__var_names[__integer_pack(__var_names.size())])...
  );
}
```

`autodiff_grad` is implemented as an expression macro. This allows us to harvest the values of the independent variables from their names, because the expression macro is expanded in the scope of the call site. It can create any meta objects (which we double-underscore to avoid shadowing independent variable names during subsequent name lookup), but may only emit a single real _return-statement_. The argument of the return is inlined into the calling expression.

The expression macro constructs the autodiff tape (the IR) with a compile-time call to `ad_builder_t::process`. This function is implemented in `libapex.so`, so `-M libapex.so` must be specified as a `circle` argument to load this shared object library as a dependency. The expression macro returns a call to the metafunction `autodiff_eval`, passing the AD builder meta object and the value of each independent variable as arguments.

```cpp
template<typename... args_t>
@meta std::array<double, sizeof...(args_t)> autodiff_eval(
  @meta const ad_builder_t& ad_builder, args_t... args) {

  @meta size_t num_vars = ad_builder.var_names.size();
  @meta size_t num_items = ad_builder.tape.size();

  // Compute the values for the whole tape. This is the forward-mode pass. 
  // It propagates values from the terminals (independent variables) through
  // the subexpressions and up to the root of the function.

  // Copy the values of the independent variables into the tape.
  double tape_values[num_items] { args... };

  // Evaluate the subexpressions.
  @meta for(size_t i = num_vars; i < num_items; ++i)
    tape_values[i] = autodiff_expr(ad_builder.tape[i].val.get());

  // Evaluate the gradients. This is a top-down reverse-mode traversal of 
  // the autodiff DAG. The partial derivatives are parsed along edges, starting
  // from the root and towards each terminal. When a terminal is visited, the
  // corresponding component of the gradient is incremented by the product of
  // all the partial derivatives from the root of the DAG down to that 
  // terminal.

  double coef[num_items];
  std::array<double, num_vars> grad { };

  // Visit each child of the root node.
  @meta int root = num_items - 1;
  @meta for(const auto& g : ad_builder.tape[root].grads) {
    // Evaluate the coefficient into the stack.
    coef[root] = autodiff_expr(g.coef.get());

    // Recurse on the child.
    @macro autodiff_tape(g.index, root);
  }

  return std::move(grad);
}
```

`autodiff_eval` is a [metafunction](https://github.com/seanbaxter/circle#metafunctions). This establishes a new scope separate from the expression context of the call site, allowing us to spread our legs a bit and declare all of the meta and real objects we care to construct. Here we allocate an array of values to hold the state of the _tape_, meaning the values of each independent variable and each subexpression encountered in the formula. These are computed bottom-up just once, and memoized into `tape_values`. 

Although the values in the tape will be used again during the top-down gradient pass, their storage may be a performance limiter in problems with a very large number of temporary nodes. Because the library defines its own IR and scheduling intelligence, it's feasible to extend the IR and emit instructions to rematerialize temporary values to alleviate storage pressure. 

## The autodiff IR

The autodiff IR needs to be comprehensive enough to encode any operations found in the expression to differentiate. We chose the design for easy lowering using intrinsics like `@op` and `@expression` to generate code from strings.

[**autodiff.hxx**](../include/apex/autodiff.hxx)
```cpp
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
```

The autodiff code in `libapex.so` generates `ad_t` trees into the tape data structure. Each tree node is allocated on the heap and stored in an `std::unique_ptr`. Because the shared object is loaded into the address space of the compiler, the result object of the foreign-function library call is fully accessible to meta code in the translation unit by way of the Circle interpreter. 

[**autodiff_codegen.hxx**](../include/apex/autodiff_codegen.hxx)
```cpp
@macro auto autodiff_expr(const ad_t* ad) {
  @meta+ if(const auto* tape = ad->as<ad_tape_t>()) {
    @emit return tape_values[tape->index];

  } else if(const auto* literal = ad->as<ad_literal_t>()) {
    @emit return literal->x;

  } else if(const auto* unary = ad->as<ad_unary_t>()) {
    @emit return @op(
      unary->op, 
      autodiff_expr(unary->a.get())
    );

  } else if(const auto* binary = ad->as<ad_binary_t>()) {
    @emit return @op(
      binary->op, 
      autodiff_expr(binary->a.get()), 
      autodiff_expr(binary->b.get())
    );

  } else if(const auto* func = ad->as<ad_func_t>()) {
  	// Support 1- and 2-parameter function calls.
    if(1 == func->args.size()) {
      @emit return @expression(func->f)(autodiff_expr(func->args[0].get()));

    } else if(2 == func->args.size()) {
      @emit return @expression(func->f)(autodiff_expr(func->args[0].get()),
        autodiff_expr(func->args[1].get()));
    }
  }
}
```

The expression macro `autodiff_expr` recurses an `ad_t` tree and switches on each node kind. 

* The macro is expanded in the scope of the caller, so the `tape_values` object is visible; this provides access to the value of each subexpression.
* The unary and binary nodes hold strings with the operator names, such as "+" or "/". We can pass these strings to `@op` along with the expression arguments to synthesize the corresponding kind of expression.
* Function call nodes have the _name_ of the function stored as a string. When evaluated with `@expression`, name lookup is performed on the qualified name (eg, "std::cos") and returns a function lvalue or overload set.

Each tape item (corresponding to sparse matrix row) includes one `ad_t` tree that renders the value of the subexpression, and one `ad_t` per child node in the DAG to compute partial derivatives. The values are computed in bottom-up order (forward through the tape), and the partial derivatives are computed in top-down order (reverse mode through the tape). An optimization potential may be exposed by evaluating all partial derivatives in parallel (there are no data dependencies between them), and using a parallelized sparse back-propagation code to concatenate the partial derivatives. Again, these choices should be made by the intelligence of the library, which is well-separated from the metaprogramming concerns of the code generator.
