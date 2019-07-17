#include <json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <apex/autodiff_codegen.hxx>

// Parse the JSON file and keep it open in j.
using nlohmann::json;
using apex::sq;

struct vec3_t {
  double x, y, z;
};

// Record the function names encountered in here!
@meta std::vector<std::string> func_names;

@macro void gen_functions(const char* filename) {
  // Open this file at compile time and parse as JSON.
  @meta std::ifstream json_file(filename);
  @meta json j;
  @meta json_file>> j;
  
  @meta for(auto& item : j.items()) {
    // For each item in the json...
    @meta std::string name = item.key();
    @meta std::string f = item.value();
    @meta std::cout<< "Injecting '"<< name<< "' : '"<< f<< "' from "<< 
      filename<< "\n";
    
    // Generate a function from the expression.
    extern "C" double @("f_" + name)(vec3_t v) {
      double x = v.x, y = v.y, z = v.z;
      return @expression(f);
    }
    
    // Generate a function to return the gradient.
    extern "C" vec3_t @("grad_" + name)(vec3_t v) {
      return apex::autodiff_grad(f.c_str(), v);
    }

    @meta func_names.push_back(name);
  }
}

@macro gen_functions("formula.json");

std::pair<double, vec3_t> eval(const char* name, vec3_t v) {
  @meta for(const std::string& f : func_names) {
    if(!strcmp(name, @string(f))) {
      return {
        @("f_" + f)(v),
        @("grad_" + f)(v)
      };
    }
  }

  printf("Unknown function %s\n", name);
  exit(1);
}

void print_usage() {
  printf("  Usage: grad2 name x y z\n");
  exit(1);
}

int main(int argc, char** argv) {
  if(5 != argc)
    print_usage();
 
  const char* f = argv[1];
  double x = atof(argv[2]);
  double y = atof(argv[3]);
  double z = atof(argv[4]);
  vec3_t v { x, y, z };

  auto result = eval(f, v);
  double val = result.first;
  vec3_t grad = result.second;

  printf("  f: %f\n", val);
  printf("  grad: { %f, %f, %f }\n", grad.x, grad.y, grad.z);
  
  return 0;
}