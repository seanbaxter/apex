#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include "parse.hxx"
#include "autodiff.hxx"

int main() {
  std::string s = "x + 3.1* z * x / tan(y / x + z)";
  auto p = apex::parse_expression(s.c_str());

  using 
  ad_builder_t ad_builder;
  ad_builder.text = s;
  ad_builder.var_names.push_back("x");
  ad_builder.var_names.push_back("y");
  ad_builder.var_names.push_back("z");
  ad_builder.tape.resize(3);

  ad_builder.recurse(p.root.get());

  return 0;
}
