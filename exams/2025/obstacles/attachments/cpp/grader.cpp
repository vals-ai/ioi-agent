#include "obstacles.h"
#include <cassert>
#include <cstdio>

int main() {
  int N, M;
  assert(2 == scanf("%d %d", &N, &M));
  std::vector<int> T(N), H(M);
  for (int i = 0; i < N; i++)
    assert(1 == scanf("%d", &T[i]));
  for (int i = 0; i < M; i++)
    assert(1 == scanf("%d", &H[i]));

  int Q;
  assert(1 == scanf("%d", &Q));
  std::vector<int> L(Q), R(Q), S(Q), D(Q);
  for (int i = 0; i < Q; i++)
    assert(4 == scanf("%d %d %d %d", &L[i], &R[i], &S[i], &D[i]));

  fclose(stdin);

  std::vector<bool> A(Q);

  initialize(T, H);

  for (int i = 0; i < Q; i++)
    A[i] = can_reach(L[i], R[i], S[i], D[i]);

  for (int i = 0; i < Q; i++)
    if (A[i])
      printf("1\n");
    else
      printf("0\n");
  fclose(stdout);

  return 0;
}
