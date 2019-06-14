#include "util.hxx"

BEGIN_APEX_NAMESPACE

// Encodes ucs into the UTF-8 buffer at s. Returs the number of characters
// encoded. 0 indicates error.
int to_utf8(char* s, int ucs) {
  if(ucs <= 0x007f) {
    s[0] = (char)ucs;
    return 1;

  } else if(ucs <= 0x07ff) {
    s[0] = 0xc0 | (ucs>> 6);
    s[1] = 0x80 | (0x3f & ucs);
    return 2;

  } else if(ucs <= 0xffff) {
    s[0] = 0xe0 | (ucs>> 12);
    s[1] = 0x80 | (0x3f & (ucs>> 6));
    s[2] = 0x80 | (0x3f & ucs);
    return 3;

  } else if(ucs <= 0x10ffff) {
    s[0] = 0xf0 | (ucs>> 18);
    s[1] = 0x80 | (0x3f & (ucs>> 12));
    s[2] = 0x80 | (0x3f & (ucs>> 6));
    s[3] = 0x80 | (0x3f & ucs);
    return 4;
  }
  return 0;
}

// Returns the number of code-units consumed and the value of the character.
// 0 indicates error.
std::pair<int, int> from_utf8(const char* s) {
  std::pair<int, int> result { };

  if(0 == (0x80 & s[0])) {
    result = std::make_pair(1, s[0]);

  } else if(0xc0 == (0xe0 & s[0])) {
    if(0x80 == (0xc0 & s[1])) {
      int ucs = (0x3f & s[1]) + ((0x1f & s[0])<< 6);
      result = std::make_pair(2, ucs);
    }

  } else if(0xe0 == (0xf0 & s[0])) {
    if(0x80 == (0xc0 & s[1]) && 
       0x80 == (0xc0 & s[2])) {
      int ucs = (0x3f & s[2]) + ((0x3f & s[1])<< 6) + ((0x0f & s[0])<< 12);
      result = std::make_pair(3, ucs);
    }

  } else if(0xf0 == (0xf8 & s[0])) {
    if(0x80 == (0xc0 & s[1]) && 
       0x80 == (0xc0 & s[2]) && 
       0x80 == (0xc0 & s[3])) {
      int ucs = (0x3f & s[3]) + ((0x3f & s[2])<< 6) + 
        ((0x3f & s[1])<< 12) + ((0x07 & s[0])<< 18);
      result = std::make_pair(4, ucs);
    }
  }
  return result;
}

END_APEX_NAMESPACE
