#include "worldmap.h"

std::vector<std::vector<int>> create_map(int N, int M, std::vector<int> A, std::vector<int> B) {

  std::vector<std::vector<int>> ans(2 * N, std::vector<int>(2 * N, 1));
  if (M > 0) {
    ans[0][0] = A[0];
    ans[0][1] = B[0];
  }

  return ans;
}
