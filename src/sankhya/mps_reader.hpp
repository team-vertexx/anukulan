#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

struct MpsOptions {
  // MPS files disagree about what an UP bound with a negative value means when no
  // lower bound has been given for that column. CPLEX documents that the lower
  // bound becomes -infinity; HiGHS leaves it at the default of 0. We default to
  // HiGHS because HiGHS is what our results are checked against, and we warn
  // whenever a file actually hits the case so it can never bite silently.
  enum class NegativeUpBound { kKeepLower, kMinusInfinity };
  NegativeUpBound negative_up_bound = NegativeUpBound::kKeepLower;

  // Values at or beyond this magnitude in BOUNDS are read as infinite.
  double infinity_threshold = kFileInf;
};

struct MpsReadResult {
  bool ok = false;
  std::string error;
  std::vector<std::string> warnings;
  Model model;

  // Counts kept for the reader's own reporting and for the benchmark harness.
  Int free_rows_dropped = 0;
  Int objective_nonzeros = 0;
};

// Reads MPS, and the QPS extension (QUADOBJ / QMATRIX) used by the
// Maros-Meszaros convex QP set. Free-format tokenisation: names may not contain
// spaces, which holds for every benchmark set we target.
MpsReadResult read_mps(const std::string& path, const MpsOptions& options = {});

MpsReadResult read_mps_stream(std::istream& in, const std::string& source_label,
                              const MpsOptions& options = {});

}  // namespace sankhya
