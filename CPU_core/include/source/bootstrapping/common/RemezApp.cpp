#include "RemezApp.h"

void testReLU(size_t deg) {
  RR maxerr;
   
  long prec = 50;
  long iteration = 200;
  long RRprec;
  int num;
  RR sc;
   
  long type = 1;
  RR scale = RR(1);

  size_t inter_num = 1;
  vector<RR> inter_start, inter_end;
  vector<RR> coeff, chebcoeff;
  inter_start.emplace_back(RR(-1));
  inter_end.emplace_back(RR(1));

  num = 1000;
   
  RRprec = 300;
   
  RR::SetPrecision(RRprec);
  sc = RR(2.0) / RR(num);

  Remez rm(ReLU, inter_num, inter_start, inter_end, sc, prec, deg, iteration, type, scale, true);
  rm.test();
  maxerr = rm.getmaxerr();

  rm.getcoeff(chebcoeff);
  Polynomial poly(deg, chebcoeff, "cheb");
  poly.cheb_to_power();
  poly.get_coeff(coeff);
   
  ofstream out1("../result/ReLUpower.txt"), out2("../result/ReLUcheb.txt");
  for (auto num : coeff) out1 << num << endl;
  for (auto num : chebcoeff) out2 << num << endl;

  RR max = RR(0);
  for (RR x = RR(-1); x < RR(1); x += RR(0.0001))
    if (abs(poly.evaluate(x) - ReLU(x)) > max) max = abs(poly.evaluate(x) - ReLU(x));
  
  max = RR(0);
  for (RR x = RR(-1); x < RR(1); x += RR(0.0001))
    if (abs(poly.evaluate_cheb(x) - ReLU(x)) > max) max = abs(poly.evaluate_cheb(x) - ReLU(x));
  
}
RR GetError(int d, RR t, bool is_first_function, int type, RR scale) {
  Remez *rm;
  RR maxerr;
  long prec = 40;
  long iteration = 20;
  long RRprec;
  int num;
  RR sc;

  size_t inter_num = 2;
  vector<RR> inter_start, inter_end;
  if (is_first_function == true) {
    inter_start.emplace_back(RR(-1));
    inter_start.emplace_back(RR(1) - t);
    inter_end.emplace_back(RR(-1) + t);
    inter_end.emplace_back(RR(1));
  } else {
    inter_start.emplace_back(RR(-1) - t);
    inter_start.emplace_back(RR(1) - t);
    inter_end.emplace_back(RR(-1) + t);
    inter_end.emplace_back(RR(1) + t);
  }

  num = 500;
  RRprec = 1500;   

  RR::SetPrecision(RRprec);
  sc = t / static_cast<RR>(num);
   
  rm = new Remez(sgn, inter_num, inter_start, inter_end, sc, prec, d + 1, iteration, type, scale, true);
  rm->test();
  maxerr = rm->getmaxerr();

  delete rm;
  return maxerr;
}
 
RR GetErrorCoeff(int d, RR t, vector<RR> &coeff, bool is_first_function, int type, RR scale) {
  Remez *rm;
  RR maxerr;
  long prec = 40;
  long iteration = 25;
  long RRprec;
  int num;
  
  RR sc;

  size_t inter_num = 2;
  vector<RR> inter_start, inter_end;
  if (is_first_function == true) {
    inter_start.emplace_back(RR(-1));
    inter_start.emplace_back(RR(1) - t);
    inter_end.emplace_back(RR(-1) + t);
    inter_end.emplace_back(RR(1));
  } else {
    inter_start.emplace_back(RR(-1) - t);
    inter_start.emplace_back(RR(1) - t);
    inter_end.emplace_back(RR(-1) + t);
    inter_end.emplace_back(RR(1) + t);
  }

  num = 1000;
  RRprec = 300;

  RR::SetPrecision(RRprec);
  sc = t / static_cast<RR>(num);
   
  rm = new Remez(sgn, inter_num, inter_start, inter_end, sc, prec, d + 1, iteration, type, scale, true);
  rm->test();
  rm->getcoeff(coeff);
  maxerr = rm->getmaxerr();

  delete rm;
  return maxerr;
}
