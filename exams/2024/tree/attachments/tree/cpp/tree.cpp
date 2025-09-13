#include "tree.h"

int n;
std::vector<int> p, w;

void init(std::vector<int> P, std::vector<int> W) {
  p = P;
  w = W;
  n = (int)p.size();
}

long long query(int L, int R) {
  return n * (long long)(R - L);
}
