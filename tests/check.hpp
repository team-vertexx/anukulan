// Minimal test scaffolding. No external dependency on purpose - the whole point of
// this project is that the stack is ours, and a test runner is not worth an exception.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace sankhya_test {

inline int& failures() {
  static int n = 0;
  return n;
}

inline void report(const char* file, int line, const std::string& what) {
  std::fprintf(stderr, "FAIL %s:%d  %s\n", file, line, what.c_str());
  ++failures();
}

inline bool close(double a, double b, double tol) {
  if (std::isinf(a) || std::isinf(b)) return a == b;
  const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
  return std::fabs(a - b) <= tol * scale;
}

inline int finish(const char* name) {
  if (failures() == 0) {
    std::printf("ok   %s\n", name);
    return 0;
  }
  std::printf("FAIL %s (%d failures)\n", name, failures());
  return 1;
}

}  // namespace sankhya_test

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) sankhya_test::report(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_EQ(a, b)                                                          \
  do {                                                                          \
    const auto va_ = (a);                                                       \
    const auto vb_ = (b);                                                       \
    if (!(va_ == vb_))                                                          \
      sankhya_test::report(__FILE__, __LINE__,                                  \
                           std::string(#a " == " #b " : got ") +                \
                               std::to_string(va_) + " vs " +                   \
                               std::to_string(vb_));                            \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
  do {                                                                          \
    const double va_ = static_cast<double>(a);                                  \
    const double vb_ = static_cast<double>(b);                                  \
    if (!sankhya_test::close(va_, vb_, (tol)))                                  \
      sankhya_test::report(__FILE__, __LINE__,                                  \
                           std::string(#a " ~= " #b " : got ") +                \
                               std::to_string(va_) + " vs " +                   \
                               std::to_string(vb_));                            \
  } while (0)

#define CHECK_STR_EQ(a, b)                                                      \
  do {                                                                          \
    const std::string va_ = (a);                                                \
    const std::string vb_ = (b);                                                \
    if (va_ != vb_)                                                             \
      sankhya_test::report(__FILE__, __LINE__,                                  \
                           std::string(#a " == " #b " : got \"") + va_ +        \
                               "\" vs \"" + vb_ + "\"");                        \
  } while (0)
