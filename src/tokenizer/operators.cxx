#include "tokenizer.hxx"
#include <cstring>
#include <algorithm>

BEGIN_APEX_NAMESPACE

namespace tok {

struct tk_symbol_t {
  const char* symbol;
  tk_kind_t kind;
};

const tk_symbol_t tk_op_symbols[] = {
  // Standard tokens.
  tk_symbol_t { "&"                 , tk_sym_amp              },
  tk_symbol_t { "&&"                , tk_sym_ampamp           },
  tk_symbol_t { "&="                , tk_sym_ampeq            },
  tk_symbol_t { "->"                , tk_sym_arrow            },
  tk_symbol_t { "->*"               , tk_sym_arrowstar        },
  tk_symbol_t { "[["                , tk_sym_attrib_l         },
  tk_symbol_t { "!"                 , tk_sym_bang             },
  tk_symbol_t { "!="                , tk_sym_bangeq           },
  tk_symbol_t { "{"                 , tk_sym_brace_l          },
  tk_symbol_t { "}"                 , tk_sym_brace_r          },
  tk_symbol_t { "["                 , tk_sym_bracket_l        },
  tk_symbol_t { "]"                 , tk_sym_bracket_r        },
  tk_symbol_t { "^"                 , tk_sym_caret            },
  tk_symbol_t { "^="                , tk_sym_careteq          },
  // tk_symbol_t { "<<<"               , tk_sym_chevron_l        },
  // tk_symbol_t { ">>>"               , tk_sym_chevron_r        },
  tk_symbol_t { ":"                 , tk_sym_col              },
  tk_symbol_t { "::"                , tk_sym_colcol           },
  tk_symbol_t { ","                 , tk_sym_comma            },
  tk_symbol_t { "."                 , tk_sym_dot              },
  tk_symbol_t { ".*"                , tk_sym_dotstar          },
  tk_symbol_t { "..."               , tk_sym_ellipsis         },
  tk_symbol_t { "="                 , tk_sym_eq               },
  tk_symbol_t { "=="                , tk_sym_eqeq             },
  tk_symbol_t { ">"                 , tk_sym_gt               },
  tk_symbol_t { ">="                , tk_sym_gteq             },
  tk_symbol_t { ">>"                , tk_sym_gtgt             },
  tk_symbol_t { ">>="               , tk_sym_gtgteq           },
  tk_symbol_t { "<"                 , tk_sym_lt               },
  tk_symbol_t { "<="                , tk_sym_lteq             },
  tk_symbol_t { "<<"                , tk_sym_ltlt             },
  tk_symbol_t { "<<="               , tk_sym_ltlteq           },
  tk_symbol_t { "-"                 , tk_sym_minus            },
  tk_symbol_t { "-="                , tk_sym_minuseq          },
  tk_symbol_t { "--"                , tk_sym_minusminus       },
  tk_symbol_t { "("                 , tk_sym_paren_l          },
  tk_symbol_t { ")"                 , tk_sym_paren_r          },
  tk_symbol_t { "%"                 , tk_sym_percent          },
  tk_symbol_t { "%="                , tk_sym_percenteq        },
  tk_symbol_t { "|"                 , tk_sym_pipe             },
  tk_symbol_t { "|="                , tk_sym_pipeeq           },
  tk_symbol_t { "||"                , tk_sym_pipepipe         },
  tk_symbol_t { "+"                 , tk_sym_plus             },
  tk_symbol_t { "+="                , tk_sym_pluseq           },
  tk_symbol_t { "++"                , tk_sym_plusplus         },
  tk_symbol_t { "?"                 , tk_sym_question         },
  tk_symbol_t { ";"                 , tk_sym_semi             },
  tk_symbol_t { "/"                 , tk_sym_slash            },
  tk_symbol_t { "/="                , tk_sym_slasheq          },
  tk_symbol_t { "*"                 , tk_sym_star             },
  tk_symbol_t { "*="                , tk_sym_stareq           },
  tk_symbol_t { "~"                 , tk_sym_tilde            },
};
const size_t num_op_symbols = sizeof(tk_op_symbols) / sizeof(tk_symbol_t);

////////////////////////////////////////////////////////////////////////////////

typedef std::pair<size_t, size_t> pair_t;

class match_operator_t {
public:
  match_operator_t();
  result_t<tk_kind_t> substring(range_t range) const;

private:
  // Return the range of operators matching the first character.
  pair_t first_char(size_t c) const;

  // Return the range of operators matching a subsequent character.
  pair_t next_char(pair_t pair, int pos, char c) const;

  std::vector<const char*> tokens;
  std::vector<tk_kind_t> kinds;
  std::vector<size_t> first_char_map;
};

match_operator_t::match_operator_t() {
  std::vector<tk_symbol_t> symbols(tk_op_symbols, 
    tk_op_symbols + num_op_symbols);
  auto cmp = [](tk_symbol_t a, tk_symbol_t b) {
    return strcmp(a.symbol, b.symbol) < 0;
  };
  std::sort(symbols.begin(), symbols.end(), cmp);

  tokens.resize(num_op_symbols);
  kinds.resize(num_op_symbols);
  for(size_t i = 0; i < num_op_symbols; ++i) {
    tokens[i] = symbols[i].symbol;
    kinds[i] = symbols[i].kind;
  }

  first_char_map.resize(257);
  for(size_t i = 0; i < 256; ++i) {
    auto cmp = [](const char* p, char c) {
      return (uint8_t)p[0] < (uint8_t)c;
    };

    auto it = std::lower_bound(tokens.begin(), tokens.end(), (char)i, cmp);
    first_char_map[i] = it - tokens.begin();
  }
  first_char_map[256] = tokens.size();
}

inline pair_t match_operator_t::first_char(size_t c) const {
  // Build a map so the range of the first character is a direct lookup.
  size_t begin = first_char_map[c];
  size_t end = first_char_map[c + 1];
  return std::make_pair(begin, end);
}

inline pair_t match_operator_t::next_char(pair_t pair, int pos, char c) const {
  assert('\0' != c);

  // Scan from left-to-right until we hit a match.
  size_t begin2 = pair.first;
  while(begin2 != pair.second && c != tokens[begin2][pos])
    ++begin2;

  // Scan from left-to-right until we hit a miss.
  size_t end2 = begin2;
  while(end2 != pair.second && c == tokens[end2][pos])
    ++end2;

  return std::make_pair(begin2, end2);
}

result_t<tk_kind_t> match_operator_t::substring(range_t range) const {
  const char* begin = range.begin;
  result_t<tk_kind_t> result;

  pair_t match = first_char((uint8_t)range.peek());
  if(match.first < match.second) {
    int pos = 0;
    pair_t match2 = match;
    while(match2.first < match2.second && range) {
      ++pos;
      ++range.begin;
      
      match = match2;
      if(char c = range.peek())
        match2 = next_char(match, pos, c);
      else
        match2.first = match2.second;
    }

    assert(match.first < match.second);
    if('\0' == tokens[match.first][pos]) {
      // We've run out of matches. Test the first operator from the input
      // range.
      result = make_result(begin, range.begin, kinds[match.first]);
    }
  }

  return result;
}

result_t<tk_kind_t> match_operator(range_t range) {
  static match_operator_t match;
  return match.substring(range);
}

} // namespace tok

END_APEX_NAMESPACE
