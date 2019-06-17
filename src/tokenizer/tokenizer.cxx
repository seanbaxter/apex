#include <apex/tokenizer.hxx>
#include <apex/parse.hxx>
#include <algorithm>

BEGIN_APEX_NAMESPACE

namespace tok {

apex::parse::range_t tokenizer_t::token_range() const {
  return { tokens.data(), tokens.data() + tokens.size() };
} 

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

void tokenizer_t::tokenize() {
  // Mark the byte of each line offset.
  size_t len = text.size();
  line_offsets.push_back(0);
  for(size_t i = 0; i < len; ++i) {
    if('\n' == text[i])
      line_offsets.push_back(i);
  }
  line_offsets.push_back(len);

  lexer_t lexer(*this);
  range_t range { text.data(), text.data() + text.size() };

  while(true) {
    // Skip past whitespace and comments.
    lexer.advance_skip(range);

    if(auto token = lexer.token(range)) {
      range.advance(token);
      tokens.push_back(token->attr);
    } else
      break;
  }
}

int tokenizer_t::token_offset(source_loc_t loc) const {
  return tokens[loc.index].begin - text.c_str();
}

int tokenizer_t::token_line(int offset) const {
  // Binary search to find the line for this byte offset.
  auto it = std::upper_bound(line_offsets.begin(), line_offsets.end(), offset);
  int line = it - line_offsets.begin() - 1;
  return line;
}

int tokenizer_t::token_col(int offset, int line) const {
  // Walk forward, decoding UTF-8 and count the column adjustment to reach 
  // the offset from the line offset.
  int col = 0;
  int pos = line_offsets[line];
  while(pos < offset) {
    std::pair<int, int> ucs = from_utf8(text.data() + pos);

    // Advance by the number of bytes in the character.
    pos += ucs.first;

    // Advance by one column.
    ++col;
  }
  return col;
}

std::pair<int, int> tokenizer_t::token_linecol(int offset) const {
  int line = token_line(offset);
  int col = token_col(offset, line);
  return { line, col };
}

std::pair<int, int> tokenizer_t::token_linecol(source_loc_t loc) const {
  return token_linecol(token_offset(loc));
}

} // namespace tok

END_APEX_NAMESPACE

