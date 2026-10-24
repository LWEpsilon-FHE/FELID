#pragma once

#include <NTL/RR.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "MinicompFunc.h"

using namespace std;
using namespace NTL;
using namespace minicomp;

enum class evaltype : int {
   
  none = 0,

  oddbaby = 1,

  baby = 2,
};

namespace minicomp {
class Tree {
 public:
  int depth;
  evaltype type;
  vector<int> tree;
  int m;
  int l;
  int b;

 public:
  Tree();
  Tree(evaltype ty);
  Tree(Tree a, Tree b, int g);
  void clear();
  
  void merge(Tree a, Tree b, int g);
};

class Polynomial {
 public:
  vector<RR> coeff;
  long deg;
  vector<RR> chebcoeff;

  Polynomial();
  Polynomial(long _deg);
  Polynomial(long _deg, vector<RR> _coeff, string tag);
  Polynomial(long _deg, RR *_coeff, string tag);

  void get_coeff(vector<RR> &_coeff);

  void chebround(Polynomial &poly, long n);
  void copy(Polynomial poly);
  void power_to_cheb();
  void cheb_to_power();
  void power_to_cheb_scale(RR scale);
  void cheb_to_power_scale(RR scale);
  RR evaluate(RR input);
  RR evaluate_cheb(RR input);
};
void mul(Polynomial &rtn, Polynomial &a, Polynomial &b);
void mul(Polynomial &rtn, Polynomial &a, RR b);
void add(Polynomial &rtn, Polynomial &a, Polynomial &b);
void subt(Polynomial &rtn, Polynomial &a, Polynomial &b);

void mulinplace(Polynomial &a, Polynomial &b);
void addinplace(Polynomial &a, Polynomial &b);
void subtinplace(Polynomial &a, Polynomial &b);

void divide_poly(Polynomial &quotient, Polynomial &remainder, Polynomial &target, Polynomial &divider);
void chebyshev(Polynomial &rtn, long deg);
void chebyshev_scale(Polynomial &rtn, long deg, RR scale);

void power_to_cheb_int(Polynomial &q, int type, RR scale);
void cheb_to_power_int(Polynomial &qround, int type, RR scale);
void eval_divide(Polynomial &pround, Polynomial &qround, Polynomial &rround, Polynomial &Ti, int type, RR scale);
void geneTi(Polynomial &Ti, int deg, int type, RR scale);

void poly_decompose_integrate(int deg, int input_type, RR input_scale, vector<RR> &coeff, Tree &tree, evaltype eval_type, int output_type, RR output_scale, ofstream &out_decomp_coeff);
void print_text_chebcoeff(ofstream &out, Polynomial &q);

}
