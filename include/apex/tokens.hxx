#pragma once
#include "util.hxx"

BEGIN_APEX_NAMESPACE

enum tk_kind_t : uint8_t {
  tk_none = 0,
  tk_ident,
  tk_int,
  tk_float,
  tk_char,
  tk_string,
  tk_kw_false, 
  tk_kw_true,
  tk_sym_amp,
  tk_sym_ampamp,
  tk_sym_ampeq,
  tk_sym_arrow,
  tk_sym_arrowstar,
  tk_sym_at,
  tk_sym_attrib_l,
  tk_sym_bang,
  tk_sym_bangeq,
  tk_sym_brace_l,
  tk_sym_brace_r,
  tk_sym_bracket_l,
  tk_sym_bracket_r,
  tk_sym_caret,
  tk_sym_careteq,
  tk_sym_chevron_l,
  tk_sym_chevron_r,
  tk_sym_col,
  tk_sym_colcol,
  tk_sym_comma,
  tk_sym_dot,
  tk_sym_dotstar,
  tk_sym_ellipsis,
  tk_sym_eq,
  tk_sym_eqeq,
  tk_sym_gt,
  tk_sym_gteq,
  tk_sym_gtgt,
  tk_sym_gtgteq,
  tk_sym_hash,
  tk_sym_hashhash,
  tk_sym_lt,
  tk_sym_lteq,
  tk_sym_ltlt,
  tk_sym_ltlteq,
  tk_sym_minus,
  tk_sym_minuseq,
  tk_sym_minusminus,
  tk_sym_paren_l,
  tk_sym_paren_r,
  tk_sym_percent,
  tk_sym_percenteq,
  tk_sym_pipe,
  tk_sym_pipeeq,
  tk_sym_pipepipe,
  tk_sym_plus,
  tk_sym_pluseq,
  tk_sym_plusplus,
  tk_sym_question,
  tk_sym_semi,
  tk_sym_slash,
  tk_sym_slasheq,
  tk_sym_star,
  tk_sym_stareq,
  tk_sym_tilde,
};

struct token_t {
  tk_kind_t kind : 8;
  int store : 24;
  const char* begin, *end;

  operator tk_kind_t() const { return kind; }
};
typedef const token_t* token_it;

// Index of the token within the token stream.
struct source_loc_t {
  int index;
};

END_APEX_NAMESPACE
