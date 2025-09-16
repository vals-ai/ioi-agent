#include "message.h"

void send_message(std::vector<bool> M, std::vector<bool> C) {
  std::vector<bool> A(31, 0);
  send_packet(A);
}

std::vector<bool> receive_message(std::vector<std::vector<bool>> R) {
  return std::vector<bool>({0, 1, 1, 0});
}
