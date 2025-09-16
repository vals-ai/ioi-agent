#include "sphinx.h"

std::vector<int> find_colours(int N, std::vector<int> X, std::vector<int> Y) {
  std::vector<int> E(N, -1);
  int x = perform_experiment(E);
  std::vector<int> G(N, 0);
  if (x == 1)
    G[0] = 1;
  return G;
}
