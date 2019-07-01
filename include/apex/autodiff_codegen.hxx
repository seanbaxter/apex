#include <apex/autodiff.hxx>
#include <array>
#include <cmath>

BEGIN_APEX_NAMESPACE

// STL doesn't include a sq function, but it's really useful, because it 
// memoizes the argument, so we don't have to evaluate its subexpression twice
// to take its square.
inline double sq(double x) {
  return x * x;
}

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
    // Evaluate and expand the arguments parameter pack.

    // TODO: Can't currently expand a parameter pack through a macro 
    // invocation. Why not? Because we expand the macro immediately so we
    // can learn its return type, which we need to continue parsing the
    // expression. But if the pack expansion is outside of the macro expansion,
    // we'll need to expand the macro prior to even parsing the expansion. 

    // Two paths forward:
    // 1) Speculatively expand the macro expansion.
    // 2) Make the macro expansion a dependent expression when its passed an
    //    unexpanded pack argument. Parse through to the end of the statement,
    //    then perform substitution and expansion. This is probably what should
    //    happen.

    // That feature will eliminate the need to switch over the
    // argument counts.
    // @emit return @expression(func->f)(
    //   autodiff_expr(func->args[__integer_pack(func->args.size())].get())...
    // );

    if(1 == func->args.size()) {
      @emit return @expression(func->f)(autodiff_expr(func->args[0].get()));

    } else if(2 == func->args.size()) {
      @emit return @expression(func->f)(autodiff_expr(func->args[0].get()),
        autodiff_expr(func->args[1].get()));
    }
  }
}

@macro void autodiff_tape(int index, int parent) {
  @meta if(index < num_vars) {
    // We've hit a terminal, which corresponds to an independent variable.
    // Increment the gradient array by the coefficient at parent.
    @member_ref(grad, index) += coef[parent];

  } else {
    // We're in a subexpression. Evaluate each of the child nodes.
    @meta for(const auto& g : autodiff.tape[index].grads) {
      // Evaluate the coefficient into the stack.
      coef[index] = coef[parent] * autodiff_expr(g.coef.get());
      @macro autodiff_tape(g.index, index);
    }
  }
}

template<typename type_t>
@meta type_t autodiff_grad(@meta const char* formula, type_t input) {

  // Parse out the names from the inputs.
  static_assert(std::is_class<type_t>::value, 
    "argument to autodiff_eval must be a class object");

  // Collect the name of each primary input.
  @meta std::vector<autodiff_var_t> vars;
  @meta size_t num_vars = @member_count(type_t);

  // Copy the values of the independent variables into the tape.
  double tape_values[num_vars];
  @meta for(int i = 0; i < num_vars; ++i) {
    // Confirm that we have a scalar double-precision term.
    static_assert(std::is_same<@member_type(type_t, i), double>::value,
      std::string("member ") + @member_name(type_t, i) + " must be type double");

    // Push the primary input name.
    @meta vars.push_back({
      @member_name(type_t, i),
      0
    });

    // Set the primary input's value.
    tape_values[i] = @member_ref(input, i);
  }

  // Construct the tape. This makes a foreign function call into libapex.so.
  @meta apex::autodiff_t autodiff = apex::make_autodiff(formula, vars);
  @meta size_t count = autodiff.tape.size();

  // Compute the values for the whole tape. This is the forward-mode pass. 
  // It propagates values from the terminals (independent variables) through
  // the subexpressions and up to the root of the function.

  // Evaluate the subexpressions.
  @meta for(size_t i = num_vars; i < count; ++i)
    tape_values[i] = autodiff_expr(autodiff.tape[i].val.get());

  // Evaluate the gradients. This is a top-down reverse-mode traversal of 
  // the autodiff DAG. The partial derivatives are parsed along edges, starting
  // from the root and towards each terminal. When a terminal is visited, the
  // corresponding component of the gradient is incremented by the product of
  // all the partial derivatives from the root of the DAG down to that 
  // terminal.
  double coef[num_vars];
  type_t grad { };

  // Visit each child of the root node.
  @meta int root = count - 1;
  @meta for(const auto& g : autodiff.tape[root].grads) {
    // Evaluate the coefficient into the stack.
    coef[root] = autodiff_expr(g.coef.get());

    // Recurse on the child.
    @macro autodiff_tape(g.index, root);
  }

  return std::move(grad);
}

END_APEX_NAMESPACE