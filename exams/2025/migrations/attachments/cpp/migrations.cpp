#include "migrations.h"

int send_message(int N, int i, int Pi) {
  if (i == 1)
    return 10;
  else if (i == 2)
    return 20;
  else if (i == 3)
    return 30;
  else
    return 0;
}

std::pair<int, int> longest_path(std::vector<int> S) {
  return {0, 3};
}
