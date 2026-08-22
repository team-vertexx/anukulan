#include <sstream>
#include <string>

#include "check.hpp"
#include "sankhya/mps_reader.hpp"

using sankhya::kInf;
using sankhya::Model;
using sankhya::MpsOptions;
using sankhya::MpsReadResult;
using sankhya::ObjSense;
using sankhya::VarType;

namespace {

MpsReadResult read(const std::string& text, const MpsOptions& opt = {}) {
  std::istringstream in(text);
  return sankhya::read_mps_stream(in, "<test>", opt);
}

int col_of(const Model& m, const std::string& name) {
  for (std::size_t j = 0; j < m.col_names.size(); ++j) {
    if (m.col_names[j] == name) return static_cast<int>(j);
  }
  return -1;
}

int row_of(const Model& m, const std::string& name) {
  for (std::size_t i = 0; i < m.row_names.size(); ++i) {
    if (m.row_names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

// A tiny model exercising every row type, used as the base for several tests.
const char* kBasic =
    "NAME          TINY\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    " G  LIM2\n"
    " E  EQ1\n"
    "COLUMNS\n"
    "    X         COST      1.0        LIM1      2.0\n"
    "    X         LIM2      3.0\n"
    "    Y         COST      -4.0       LIM1      5.0\n"
    "    Y         EQ1       6.0\n"
    "RHS\n"
    "    RHS       LIM1      10.0       LIM2      20.0\n"
    "    RHS       EQ1       30.0\n"
    "ENDATA\n";

void test_basic() {
  const MpsReadResult r = read(kBasic);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  const Model& m = r.model;
  CHECK_STR_EQ(m.name, "TINY");
  CHECK_EQ(m.num_rows(), 3);
  CHECK_EQ(m.num_cols(), 2);
  CHECK_EQ(m.constraints.nnz(), 4);
  CHECK_EQ(r.objective_nonzeros, 2);
  CHECK(m.sense == ObjSense::kMinimize);

  const int x = col_of(m, "X");
  const int y = col_of(m, "Y");
  CHECK_NEAR(m.objective[static_cast<std::size_t>(x)], 1.0, 0.0);
  CHECK_NEAR(m.objective[static_cast<std::size_t>(y)], -4.0, 0.0);

  // Default bounds: non-negative, no upper bound.
  CHECK_NEAR(m.col_lower[static_cast<std::size_t>(x)], 0.0, 0.0);
  CHECK_NEAR(m.col_upper[static_cast<std::size_t>(x)], kInf, 0.0);

  const std::size_t l1 = static_cast<std::size_t>(row_of(m, "LIM1"));
  const std::size_t l2 = static_cast<std::size_t>(row_of(m, "LIM2"));
  const std::size_t e1 = static_cast<std::size_t>(row_of(m, "EQ1"));
  CHECK_NEAR(m.row_lower[l1], -kInf, 0.0);  // L row
  CHECK_NEAR(m.row_upper[l1], 10.0, 0.0);
  CHECK_NEAR(m.row_lower[l2], 20.0, 0.0);   // G row
  CHECK_NEAR(m.row_upper[l2], kInf, 0.0);
  CHECK_NEAR(m.row_lower[e1], 30.0, 0.0);   // E row
  CHECK_NEAR(m.row_upper[e1], 30.0, 0.0);
}

void test_objective_offset_is_negated() {
  // An RHS entry on the objective row is the negated constant term. Getting the
  // sign wrong here shifts every reported objective by a constant, which is
  // exactly the kind of bug that survives until benchmark day.
  const std::string text =
      "NAME          OFF\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        2.0        COST      7.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  CHECK_NEAR(r.model.objective_offset, -7.0, 0.0);
}

void test_ranges() {
  // The sign of a RANGES value matters only for E rows. For L and G rows the
  // magnitude is used. Checked against the HiGHS reader's behaviour.
  const std::string text =
      "NAME          RNG\n"
      "ROWS\n"
      " N  COST\n"
      " L  RL\n"
      " G  RG\n"
      " E  REP\n"
      " E  REN\n"
      " E  REZ\n"
      "COLUMNS\n"
      "    X         COST      1.0        RL        1.0\n"
      "    X         RG        1.0        REP       1.0\n"
      "    X         REN       1.0        REZ       1.0\n"
      "RHS\n"
      "    RHS       RL        10.0       RG        20.0\n"
      "    RHS       REP       30.0       REN       40.0\n"
      "    RHS       REZ       50.0\n"
      "RANGES\n"
      "    RNG       RL        -4.0       RG        -5.0\n"
      "    RNG       REP       6.0        REN       -7.0\n"
      "    RNG       REZ       0.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  const Model& m = r.model;
  auto lo = [&](const char* n) {
    return m.row_lower[static_cast<std::size_t>(row_of(m, n))];
  };
  auto hi = [&](const char* n) {
    return m.row_upper[static_cast<std::size_t>(row_of(m, n))];
  };

  // L row, negative range: sign ignored, [b - |r|, b].
  CHECK_NEAR(lo("RL"), 6.0, 0.0);
  CHECK_NEAR(hi("RL"), 10.0, 0.0);
  // G row, negative range: sign ignored, [b, b + |r|].
  CHECK_NEAR(lo("RG"), 20.0, 0.0);
  CHECK_NEAR(hi("RG"), 25.0, 0.0);
  // E row, positive range: [b, b + r].
  CHECK_NEAR(lo("REP"), 30.0, 0.0);
  CHECK_NEAR(hi("REP"), 36.0, 0.0);
  // E row, negative range: [b - |r|, b].
  CHECK_NEAR(lo("REN"), 33.0, 0.0);
  CHECK_NEAR(hi("REN"), 40.0, 0.0);
  // E row, zero range: stays an equality.
  CHECK_NEAR(lo("REZ"), 50.0, 0.0);
  CHECK_NEAR(hi("REZ"), 50.0, 0.0);
}

void test_bounds() {
  const std::string text =
      "NAME          BND\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    A         COST      1.0        R1        1.0\n"
      "    B         COST      1.0        R1        1.0\n"
      "    C         COST      1.0        R1        1.0\n"
      "    D         COST      1.0        R1        1.0\n"
      "    E         COST      1.0        R1        1.0\n"
      "    F         COST      1.0        R1        1.0\n"
      "    G         COST      1.0        R1        1.0\n"
      "    H         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        1.0\n"
      "BOUNDS\n"
      " UP BND       A         4.0\n"
      " LO BND       B         -2.0\n"
      " FX BND       C         3.5\n"
      " FR BND       D\n"
      " MI BND       E\n"
      " PL BND       F\n"
      " BV BND       G\n"
      " UI BND       H         9.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  const Model& m = r.model;
  auto lo = [&](const char* n) {
    return m.col_lower[static_cast<std::size_t>(col_of(m, n))];
  };
  auto hi = [&](const char* n) {
    return m.col_upper[static_cast<std::size_t>(col_of(m, n))];
  };
  auto ty = [&](const char* n) {
    return m.col_type[static_cast<std::size_t>(col_of(m, n))];
  };

  CHECK_NEAR(lo("A"), 0.0, 0.0);
  CHECK_NEAR(hi("A"), 4.0, 0.0);
  CHECK_NEAR(lo("B"), -2.0, 0.0);
  CHECK_NEAR(hi("B"), kInf, 0.0);
  CHECK_NEAR(lo("C"), 3.5, 0.0);
  CHECK_NEAR(hi("C"), 3.5, 0.0);
  CHECK_NEAR(lo("D"), -kInf, 0.0);
  CHECK_NEAR(hi("D"), kInf, 0.0);
  CHECK_NEAR(lo("E"), -kInf, 0.0);
  CHECK_NEAR(hi("E"), kInf, 0.0);
  CHECK_NEAR(lo("F"), 0.0, 0.0);
  CHECK_NEAR(hi("F"), kInf, 0.0);
  CHECK_NEAR(lo("G"), 0.0, 0.0);
  CHECK_NEAR(hi("G"), 1.0, 0.0);
  CHECK(ty("G") == VarType::kInteger);
  CHECK_NEAR(hi("H"), 9.0, 0.0);
  CHECK(ty("H") == VarType::kInteger);
}

void test_negative_up_bound_conventions() {
  // The one convention where real solvers genuinely disagree. Default follows
  // HiGHS (lower stays at 0); the option follows CPLEX (lower becomes -inf).
  const std::string text =
      "NAME          NEGUP\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        -10.0\n"
      "BOUNDS\n"
      " UP BND       X         -3.0\n"
      "ENDATA\n";

  const MpsReadResult keep = read(text);
  // Lower stays 0 and upper is -3, which is an empty interval, so the model is
  // correctly rejected by validation rather than silently mis-solved.
  CHECK(!keep.ok);
  CHECK(!keep.warnings.empty());

  MpsOptions cplex;
  cplex.negative_up_bound = MpsOptions::NegativeUpBound::kMinusInfinity;
  const MpsReadResult minf = read(text, cplex);
  CHECK(minf.ok);
  if (minf.ok) {
    CHECK_NEAR(minf.model.col_lower[0], -kInf, 0.0);
    CHECK_NEAR(minf.model.col_upper[0], -3.0, 0.0);
  }
}

void test_integer_markers_and_free_rows() {
  const std::string text =
      "NAME          MRK\n"
      "ROWS\n"
      " N  COST\n"
      " N  FREE1\n"
      " L  R1\n"
      "COLUMNS\n"
      "    CONT      COST      1.0        R1        1.0\n"
      "    MARKER                 'MARKER'                 'INTORG'\n"
      "    INT1      COST      2.0        R1        1.0\n"
      "    INT1      FREE1     9.0\n"
      "    MARKER                 'MARKER'                 'INTEND'\n"
      "    CONT2     COST      3.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        5.0        FREE1     8.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  const Model& m = r.model;
  CHECK_EQ(r.free_rows_dropped, 1);
  CHECK_EQ(m.num_rows(), 1);  // the second N row is dropped, like HiGHS does
  CHECK_EQ(m.num_cols(), 3);
  CHECK_EQ(m.constraints.nnz(), 3);  // the FREE1 entry is not stored
  CHECK(m.col_type[static_cast<std::size_t>(col_of(m, "CONT"))] == VarType::kContinuous);
  CHECK(m.col_type[static_cast<std::size_t>(col_of(m, "INT1"))] == VarType::kInteger);
  CHECK(m.col_type[static_cast<std::size_t>(col_of(m, "CONT2"))] ==
        VarType::kContinuous);
  CHECK(m.has_integers());
}

void test_duplicate_matrix_entries_are_summed() {
  const std::string text =
      "NAME          DUP\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        2.0\n"
      "    X         R1        3.0\n"
      "RHS\n"
      "    RHS       R1        5.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  CHECK_EQ(r.model.constraints.nnz(), 1);
  CHECK_NEAR(r.model.constraints.value()[0], 5.0, 0.0);
}

void test_objsense_section() {
  const std::string text =
      "NAME          MAXP\n"
      "OBJSENSE\n"
      "    MAX\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        5.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  CHECK(r.model.sense == ObjSense::kMaximize);

  const std::string inline_text =
      "NAME          MAXI\n"
      "OBJSENSE      MAXIMIZE\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        5.0\n"
      "ENDATA\n";
  const MpsReadResult r2 = read(inline_text);
  CHECK(r2.ok);
  CHECK(r2.model.sense == ObjSense::kMaximize);
}

void test_quadratic_objective() {
  // QUADOBJ gives one triangle; the reader must mirror it into a full symmetric Q.
  const std::string text =
      "NAME          QP1\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "    Y         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        1.0\n"
      "QUADOBJ\n"
      "    X         X         2.0\n"
      "    X         Y         3.0\n"
      "    Y         Y         4.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  const Model& m = r.model;
  CHECK(m.has_hessian());
  CHECK_EQ(m.hessian.nnz(), 4);  // 2 diagonal + the mirrored off-diagonal pair
  const double x[2] = {1.0, 1.0};
  double q[2] = {0.0, 0.0};
  m.hessian.multiply(x, q);
  CHECK_NEAR(q[0], 5.0, 1e-15);  // 2 + 3
  CHECK_NEAR(q[1], 7.0, 1e-15);  // 3 + 4

  // A file that already lists both triangles must not be mirrored twice.
  const std::string both =
      "NAME          QP2\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "    Y         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        1.0\n"
      "QUADOBJ\n"
      "    X         X         2.0\n"
      "    X         Y         3.0\n"
      "    Y         X         3.0\n"
      "    Y         Y         4.0\n"
      "ENDATA\n";
  const MpsReadResult r2 = read(both);
  CHECK(r2.ok);
  if (r2.ok) {
    CHECK_EQ(r2.model.hessian.nnz(), 4);
    double q2[2] = {0.0, 0.0};
    r2.model.hessian.multiply(x, q2);
    CHECK_NEAR(q2[0], 5.0, 1e-15);
    CHECK_NEAR(q2[1], 7.0, 1e-15);
  }
}

void test_rhs_without_set_name() {
  // SIF-style files omit the RHS set name. Parity of the token count decides.
  const std::string text =
      "NAME          NOSET\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      " L  R2\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "    X         R2        1.0\n"
      "RHS\n"
      "    R1        5.0        R2        6.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  CHECK_NEAR(r.model.row_upper[0], 5.0, 0.0);
  CHECK_NEAR(r.model.row_upper[1], 6.0, 0.0);
}

void test_errors_are_reported() {
  const std::string unknown_row =
      "NAME          BAD\n"
      "ROWS\n"
      " N  COST\n"
      "COLUMNS\n"
      "    X         NOSUCHROW 1.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(unknown_row);
  CHECK(!r.ok);
  CHECK(r.error.find("NOSUCHROW") != std::string::npos);

  const std::string bad_value =
      "NAME          BAD2\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         R1        not-a-number\n"
      "ENDATA\n";
  const MpsReadResult r2 = read(bad_value);
  CHECK(!r2.ok);

  const std::string bad_type =
      "NAME          BAD3\n"
      "ROWS\n"
      " Q  WEIRD\n"
      "ENDATA\n";
  const MpsReadResult r3 = read(bad_type);
  CHECK(!r3.ok);
}

void test_dos_line_endings_and_comments() {
  const std::string text =
      "* a comment line\r\n"
      "NAME          DOS\r\n"
      "ROWS\r\n"
      " N  COST\r\n"
      " L  R1\r\n"
      "COLUMNS\r\n"
      "    X         COST      1.0        R1        2.0\r\n"
      "RHS\r\n"
      "    RHS       R1        5.0\r\n"
      "ENDATA\r\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "error: %s\n", r.error.c_str());
    return;
  }
  CHECK_STR_EQ(r.model.name, "DOS");
  CHECK_NEAR(r.model.constraints.value()[0], 2.0, 0.0);
}

void test_fortran_exponent() {
  const std::string text =
      "NAME          DEXP\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.5D+02\n"
      "RHS\n"
      "    RHS       R1        5.0\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (r.ok) CHECK_NEAR(r.model.constraints.value()[0], 150.0, 1e-12);
}

void test_large_bound_is_infinite() {
  const std::string text =
      "NAME          BIG\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      1.0        R1        1.0\n"
      "RHS\n"
      "    RHS       R1        5.0\n"
      "BOUNDS\n"
      " UP BND       X         1.0e30\n"
      "ENDATA\n";
  const MpsReadResult r = read(text);
  CHECK(r.ok);
  if (r.ok) CHECK_NEAR(r.model.col_upper[0], kInf, 0.0);
}

}  // namespace

int main() {
  test_basic();
  test_objective_offset_is_negated();
  test_ranges();
  test_bounds();
  test_negative_up_bound_conventions();
  test_integer_markers_and_free_rows();
  test_duplicate_matrix_entries_are_summed();
  test_objsense_section();
  test_quadratic_objective();
  test_rhs_without_set_name();
  test_errors_are_reported();
  test_dos_line_endings_and_comments();
  test_fortran_exponent();
  test_large_bound_is_infinite();
  return sankhya_test::finish("test_mps");
}
