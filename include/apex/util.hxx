#pragma once
#include <utility>
#include <memory>
#include <vector>
#include <cassert>

#define BEGIN_APEX_NAMESPACE namespace apex {
#define END_APEX_NAMESPACE }

BEGIN_APEX_NAMESPACE

// Encodes ucs into the UTF-8 buffer at s. Returns the number of characters
// encoded. 0 indicates error.
int to_utf8(char* s, int ucs);

// Returns the number of code-units consumed and the value of the character.
// 0 indicates error.
std::pair<int, int> from_utf8(const char* s);

////////////////////////////////////////////////////////////////////////////////
// Reusable types for result_t<> returns. Combines range and attribute.

template<typename attr_t, typename range_t>
struct result_base_t {
  result_base_t() { }
  result_base_t(range_t range, attr_t attr) :
    range(range), attr(std::move(attr)), success(true) { }

  template<typename attr2_t>
  result_base_t(result_base_t<attr2_t, range_t>&& rhs) :
    range(rhs.range), attr(std::move(rhs.attr)), success(rhs.success) { }

  range_t range;
  attr_t attr;
  bool success = false;
};

template<typename attr_t, typename range_t>
class result_template_t : protected result_base_t<attr_t, range_t> {
public:
  typedef result_base_t<attr_t, range_t> base_t;

  result_template_t() { }
  result_template_t(range_t range, attr_t attr) :
    base_t(range, std::move(attr)) { }

  template<typename attr2_t>
  result_template_t(result_template_t<attr2_t, range_t>&& rhs) :
    base_t(std::move(rhs.get_base())) { }

  explicit operator bool() const { return this->success; }

  base_t* operator->() { 
    assert(this->success);
    return this; 
  }

  const base_t* operator->() const { 
    assert(this->success); 
    return this; 
  }

  // Ugly hack to let the move constructor upcast through a protected base
  // class.
  base_t& get_base() { return *this; }
};

struct unused_t { };


END_APEX_NAMESPACE
