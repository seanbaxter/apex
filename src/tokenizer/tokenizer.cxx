#include <apex/tokenizer.hxx>

BEGIN_APEX_NAMESPACE

namespace tok {

int tokenizer_t::reg_string(range_t range) {
  int id = find_string(range);
  if(-1 == id) {
    id = (int)strings.size();
    strings.push_back(std::string(range.begin, range.end));
  }
  return id;
}

int tokenizer_t::find_string(range_t range) const {
  for(size_t i = 0; i < strings.size(); ++i) { 
    if(0 == strings[i].compare(range.begin))
      return i;
  }
  return -1;
}

std::vector<apex::token_t> tokenizer_t::tokenize(range_t range) {
  lexer_t lexer(*this);

  std::vector<apex::token_t> tokens;
  while(true) {
    // Skip past whitespace and comments.
    lexer.advance_skip(range);

    if(auto token = lexer.token(range)) {
      range.advance(token);
      tokens.push_back(token->attr);
    } else
      break;
  }

  return tokens;
}

} // namespace tok

END_APEX_NAMESPACE

