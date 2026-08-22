#include "sankhya/mps_reader.hpp"

#include <cctype>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <unordered_map>
#include <unordered_set>

namespace sankhya {
namespace {

enum class Section {
  kNone,
  kName,
  kObjSense,
  kRows,
  kColumns,
  kRhs,
  kRanges,
  kBounds,
  kQuadObj,  // one triangle given, the other implied
  kQMatrix,  // every entry given explicitly
  kEndata,
};

// How a row name resolves. Real constraints get a non-negative index.
constexpr Int kObjectiveRow = -1;
constexpr Int kFreeRow = -2;

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

std::vector<std::string> tokenize(const std::string& line) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    if (i >= line.size()) break;
    const std::size_t start = i;
    while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    out.push_back(line.substr(start, i - start));
  }
  return out;
}

// Fixed-format MPS puts each field in a fixed column range, which is how a name
// is allowed to contain spaces. The classic layout, in 1-based columns:
//
//   field 1: 2-3    field 2: 5-12   field 3: 15-22
//   field 4: 25-36  field 5: 40-47  field 6: 50-61
//
// The last field is taken to the end of the line, because writers routinely let
// a value spill a character or two past column 61.
std::vector<std::string> tokenize_fixed(const std::string& line) {
  static constexpr std::size_t kBegin[6] = {1, 4, 14, 24, 39, 49};
  static constexpr std::size_t kEnd[6] = {3, 12, 22, 36, 47,
                                          std::numeric_limits<std::size_t>::max()};
  std::vector<std::string> out;
  for (int f = 0; f < 6; ++f) {
    if (kBegin[f] >= line.size()) break;
    const std::size_t stop = std::min(kEnd[f], line.size());
    if (stop <= kBegin[f]) continue;
    std::string field = line.substr(kBegin[f], stop - kBegin[f]);
    const std::size_t first = field.find_first_not_of(" \t");
    if (first == std::string::npos) continue;
    const std::size_t last = field.find_last_not_of(" \t");
    out.push_back(field.substr(first, last - first + 1));
  }
  return out;
}

// Some old Fortran-written files use D for the exponent. Cheap insurance.
bool parse_double(const std::string& token, double* out) {
  std::string s = token;
  for (char& c : s) {
    if (c == 'd' || c == 'D') c = 'E';
  }
  const char* begin = s.c_str();
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return false;
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (*end != '\0') return false;
  if (std::isnan(v)) return false;
  *out = v;
  return true;
}

Section section_from(const std::string& word) {
  const std::string w = upper(word);
  if (w == "NAME") return Section::kName;
  if (w == "OBJSENSE" || w == "OBJSENS") return Section::kObjSense;
  if (w == "ROWS") return Section::kRows;
  if (w == "COLUMNS") return Section::kColumns;
  if (w == "RHS") return Section::kRhs;
  if (w == "RANGES") return Section::kRanges;
  if (w == "BOUNDS") return Section::kBounds;
  if (w == "QUADOBJ" || w == "QSECTION" || w == "QUADS") return Section::kQuadObj;
  if (w == "QMATRIX") return Section::kQMatrix;
  if (w == "ENDATA") return Section::kEndata;
  return Section::kNone;
}

bool is_marker_line(const std::vector<std::string>& t, bool* integer_on,
                    bool* recognised) {
  *recognised = false;
  bool has_marker = false;
  for (const std::string& tok : t) {
    const std::string w = upper(tok);
    if (w == "'MARKER'" || w == "MARKER") has_marker = true;
  }
  if (!has_marker) return false;
  for (const std::string& tok : t) {
    const std::string w = upper(tok);
    if (w == "'INTORG'" || w == "INTORG") {
      *integer_on = true;
      *recognised = true;
    } else if (w == "'INTEND'" || w == "INTEND") {
      *integer_on = false;
      *recognised = true;
    }
  }
  return true;
}

class Reader {
 public:
  Reader(MpsOptions options, std::string source)
      : options_(options), source_(std::move(source)) {}

  MpsReadResult run(std::istream& in);

 private:
  static constexpr std::size_t kMaxWarnings = 64;

  void warn(const std::string& message) {
    if (result_.warnings.size() < kMaxWarnings) {
      result_.warnings.push_back(message);
    } else if (result_.warnings.size() == kMaxWarnings) {
      result_.warnings.emplace_back("(further warnings suppressed)");
    }
  }

  bool fail(const std::string& message) {
    result_.ok = false;
    result_.error = source_ + ":" + std::to_string(line_no_) + ": " + message;
    return false;
  }

  std::vector<std::string> tokens_for(const std::string& line) const {
    return fixed_format_ ? tokenize_fixed(line) : tokenize(line);
  }

  bool flush_rows();
  bool handle_rows(const std::vector<std::string>& t);
  bool handle_columns(const std::vector<std::string>& t);
  bool handle_rhs(const std::vector<std::string>& t);
  bool handle_ranges(const std::vector<std::string>& t);
  bool handle_bounds(const std::vector<std::string>& t);
  bool handle_quadratic(const std::vector<std::string>& t, bool full_matrix);
  bool finalize();

  Int find_row(const std::string& name, bool* known) const;
  Int column_index(const std::string& name, bool create);

  MpsOptions options_;
  std::string source_;
  MpsReadResult result_;
  int line_no_ = 0;

  bool seen_objective_ = false;
  bool in_integer_marker_ = false;
  bool fixed_format_ = false;

  // ROWS lines are buffered so the format can be decided from the section as a
  // whole before any of them is interpreted. ROWS always precedes the sections
  // that depend on the chosen format, so one buffered section is enough.
  std::vector<std::string> pending_row_lines_;
  std::vector<int> pending_row_line_numbers_;

  std::unordered_map<std::string, Int> row_index_;
  std::unordered_map<std::string, Int> col_index_;

  std::vector<char> row_type_;  // 'E', 'L' or 'G'
  std::vector<double> row_rhs_;
  std::vector<bool> row_has_rhs_;
  std::vector<double> row_range_;
  std::vector<bool> row_has_range_;

  std::vector<Triplet> a_entries_;
  std::vector<Triplet> q_entries_;
  std::unordered_set<long long> q_offdiag_seen_;
  bool q_both_triangles_ = false;
  bool q_explicit_full_ = false;

  std::vector<double> objective_;
  std::vector<bool> has_lower_;
  std::vector<bool> has_upper_;

  std::size_t duplicate_bounds_ = 0;
  std::size_t negative_up_bounds_ = 0;
};

Int Reader::find_row(const std::string& name, bool* known) const {
  const auto it = row_index_.find(name);
  if (it == row_index_.end()) {
    *known = false;
    return 0;
  }
  *known = true;
  return it->second;
}

Int Reader::column_index(const std::string& name, bool create) {
  const auto it = col_index_.find(name);
  if (it != col_index_.end()) return it->second;
  if (!create) return -1;
  const Int idx = static_cast<Int>(result_.model.col_names.size());
  col_index_.emplace(name, idx);
  result_.model.col_names.push_back(name);
  objective_.push_back(0.0);
  result_.model.col_lower.push_back(0.0);
  result_.model.col_upper.push_back(kInf);
  result_.model.col_type.push_back(in_integer_marker_ ? VarType::kInteger
                                                      : VarType::kContinuous);
  has_lower_.push_back(false);
  has_upper_.push_back(false);
  return idx;
}

bool Reader::flush_rows() {
  if (pending_row_lines_.empty()) return true;

  if (options_.format == MpsOptions::Format::kFixed) {
    fixed_format_ = true;
  } else if (options_.format == MpsOptions::Format::kFree) {
    fixed_format_ = false;
  } else {
    // A free format ROWS line is exactly `type name`. Anything else means the
    // names carry spaces, which only fixed format allows.
    fixed_format_ = false;
    for (const std::string& line : pending_row_lines_) {
      if (tokenize(line).size() != 2) {
        fixed_format_ = true;
        break;
      }
    }
  }
  result_.fixed_format = fixed_format_;

  for (std::size_t i = 0; i < pending_row_lines_.size(); ++i) {
    line_no_ = pending_row_line_numbers_[i];
    if (!handle_rows(tokens_for(pending_row_lines_[i]))) return false;
  }
  pending_row_lines_.clear();
  pending_row_line_numbers_.clear();
  return true;
}

bool Reader::handle_rows(const std::vector<std::string>& t) {
  if (t.size() < 2) return fail("ROWS entry needs a type and a name");
  const std::string type = upper(t[0]);
  const std::string& name = t[1];

  if (row_index_.find(name) != row_index_.end()) {
    warn("duplicate row name \"" + name + "\": later definition ignored");
    return true;
  }

  if (type == "N") {
    // The first N row is the objective. Every later one is a free row, which
    // HiGHS drops outright - we match that so row indices agree with the
    // solver we check ourselves against.
    if (!seen_objective_) {
      seen_objective_ = true;
      row_index_.emplace(name, kObjectiveRow);
    } else {
      row_index_.emplace(name, kFreeRow);
      result_.free_rows_dropped++;
    }
    return true;
  }
  if (type != "L" && type != "G" && type != "E")
    return fail("unknown row type \"" + t[0] + "\"");

  row_index_.emplace(name, static_cast<Int>(row_type_.size()));
  row_type_.push_back(type[0]);
  row_rhs_.push_back(0.0);
  row_has_rhs_.push_back(false);
  row_range_.push_back(0.0);
  row_has_range_.push_back(false);
  result_.model.row_names.push_back(name);
  return true;
}

bool Reader::handle_columns(const std::vector<std::string>& t) {
  bool marker_recognised = false;
  if (is_marker_line(t, &in_integer_marker_, &marker_recognised)) {
    if (!marker_recognised) warn("MARKER line without INTORG/INTEND ignored");
    return true;
  }
  if (t.size() < 3 || t.size() % 2 == 0)
    return fail("COLUMNS entry must be a column name followed by name/value pairs");

  const Int col = column_index(t[0], true);
  for (std::size_t k = 1; k + 1 < t.size(); k += 2) {
    bool known = false;
    const Int row = find_row(t[k], &known);
    if (!known) return fail("unknown row name \"" + t[k] + "\" in COLUMNS");
    double value = 0.0;
    if (!parse_double(t[k + 1], &value))
      return fail("cannot parse value \"" + t[k + 1] + "\" in COLUMNS");
    if (row == kFreeRow) continue;
    if (row == kObjectiveRow) {
      objective_[sz(col)] += value;
      result_.objective_nonzeros++;
    } else {
      a_entries_.push_back(Triplet{row, col, value});
    }
  }
  return true;
}

bool Reader::handle_rhs(const std::vector<std::string>& t) {
  // A leading set name is present exactly when the token count is odd.
  const std::size_t first = t.size() % 2;
  if (t.size() < first + 2)
    return fail("RHS entry must contain at least one name/value pair");
  for (std::size_t k = first; k + 1 < t.size(); k += 2) {
    bool known = false;
    const Int row = find_row(t[k], &known);
    if (!known) return fail("unknown row name \"" + t[k] + "\" in RHS");
    double value = 0.0;
    if (!parse_double(t[k + 1], &value))
      return fail("cannot parse value \"" + t[k + 1] + "\" in RHS");
    if (row == kFreeRow) continue;
    if (row == kObjectiveRow) {
      // Convention, and what HiGHS does: an RHS on the objective row is the
      // negated constant term.
      result_.model.objective_offset = -value;
      continue;
    }
    if (row_has_rhs_[sz(row)]) warn("duplicate RHS entry for row \"" + t[k] + "\"");
    row_rhs_[sz(row)] = value;
    row_has_rhs_[sz(row)] = true;
  }
  return true;
}

bool Reader::handle_ranges(const std::vector<std::string>& t) {
  const std::size_t first = t.size() % 2;
  if (t.size() < first + 2)
    return fail("RANGES entry must contain at least one name/value pair");
  for (std::size_t k = first; k + 1 < t.size(); k += 2) {
    bool known = false;
    const Int row = find_row(t[k], &known);
    if (!known) return fail("unknown row name \"" + t[k] + "\" in RANGES");
    double value = 0.0;
    if (!parse_double(t[k + 1], &value))
      return fail("cannot parse value \"" + t[k + 1] + "\" in RANGES");
    if (row < 0) continue;
    if (row_has_range_[sz(row)]) warn("duplicate RANGES entry for row \"" + t[k] + "\"");
    row_range_[sz(row)] = value;
    row_has_range_[sz(row)] = true;
  }
  return true;
}

bool Reader::handle_bounds(const std::vector<std::string>& t) {
  if (t.empty()) return fail("empty BOUNDS entry");
  const std::string type = upper(t[0]);

  const bool takes_value = (type == "UP" || type == "LO" || type == "FX" ||
                            type == "LI" || type == "UI" || type == "SI" ||
                            type == "SC");
  const bool no_value = (type == "FR" || type == "MI" || type == "PL" || type == "BV");
  if (!takes_value && !no_value)
    return fail("unknown bound type \"" + t[0] + "\"");

  // Layout is `TYPE [set_name] column [value]`. The set name is optional, so the
  // token count decides; the one genuinely ambiguous case (a valueless bound type
  // with three tokens) is resolved by checking whether the second token names a
  // column we already know.
  std::size_t col_pos = 0;
  std::size_t value_pos = 0;
  if (takes_value) {
    if (t.size() == 3) {
      col_pos = 1;
      value_pos = 2;
    } else if (t.size() == 4) {
      col_pos = 2;
      value_pos = 3;
    } else {
      return fail("bound type " + type + " needs a column and a value");
    }
  } else {
    if (t.size() == 2) {
      col_pos = 1;
    } else if (t.size() == 3) {
      col_pos = (col_index_.find(t[1]) != col_index_.end()) ? 1 : 2;
    } else if (t.size() == 4) {
      col_pos = 2;
    } else {
      return fail("bound type " + type + " needs a column name");
    }
  }

  double value = 0.0;
  if (takes_value && !parse_double(t[value_pos], &value))
    return fail("cannot parse bound value \"" + t[value_pos] + "\"");
  if (takes_value && std::fabs(value) >= options_.infinity_threshold)
    value = (value > 0) ? kInf : -kInf;

  const Int col = column_index(t[col_pos], true);
  const std::size_t c = sz(col);
  Model& m = result_.model;

  const bool sets_lower = (type == "LO" || type == "FX" || type == "MI" ||
                           type == "LI" || type == "BV" || type == "FR");
  const bool sets_upper = (type == "UP" || type == "FX" || type == "PL" ||
                           type == "UI" || type == "BV" || type == "FR" ||
                           type == "SI" || type == "SC");

  if ((sets_lower && has_lower_[c]) || (sets_upper && has_upper_[c])) {
    duplicate_bounds_++;
    warn("duplicate bound for column \"" + t[col_pos] + "\": ignored");
    return true;
  }

  if (type == "FR") {
    m.col_lower[c] = -kInf;
    m.col_upper[c] = kInf;
  } else if (type == "MI") {
    m.col_lower[c] = -kInf;
  } else if (type == "PL") {
    m.col_upper[c] = kInf;
  } else if (type == "BV") {
    m.col_lower[c] = 0.0;
    m.col_upper[c] = 1.0;
    m.col_type[c] = VarType::kInteger;
  } else if (type == "LO") {
    m.col_lower[c] = value;
  } else if (type == "UP") {
    m.col_upper[c] = value;
    if (value < 0.0 && !has_lower_[c]) {
      negative_up_bounds_++;
      if (options_.negative_up_bound == MpsOptions::NegativeUpBound::kMinusInfinity &&
          m.col_type[c] == VarType::kContinuous) {
        m.col_lower[c] = -kInf;
        warn("negative UP bound on \"" + t[col_pos] +
             "\" with no lower bound: lower set to -infinity (CPLEX convention)");
      } else {
        warn("negative UP bound on \"" + t[col_pos] +
             "\" with no lower bound: lower kept at 0 (HiGHS convention)");
      }
    }
  } else if (type == "FX") {
    m.col_lower[c] = value;
    m.col_upper[c] = value;
  } else if (type == "LI") {
    m.col_lower[c] = value;
    m.col_type[c] = VarType::kInteger;
  } else if (type == "UI") {
    m.col_upper[c] = value;
    m.col_type[c] = VarType::kInteger;
  } else if (type == "SI") {
    m.col_upper[c] = value;
    m.col_type[c] = VarType::kSemiInteger;
  } else if (type == "SC") {
    m.col_upper[c] = value;
    m.col_type[c] = VarType::kSemiContinuous;
  }

  if (sets_lower) has_lower_[c] = true;
  if (sets_upper) has_upper_[c] = true;
  return true;
}

bool Reader::handle_quadratic(const std::vector<std::string>& t, bool full_matrix) {
  if (t.size() < 3 || t.size() % 2 == 0)
    return fail("quadratic entry must be a column name followed by name/value pairs");
  if (full_matrix) q_explicit_full_ = true;

  const Int i = column_index(t[0], true);
  for (std::size_t k = 1; k + 1 < t.size(); k += 2) {
    const Int j = column_index(t[k], true);
    double value = 0.0;
    if (!parse_double(t[k + 1], &value))
      return fail("cannot parse quadratic value \"" + t[k + 1] + "\"");
    q_entries_.push_back(Triplet{i, j, value});
    if (i != j) {
      const long long key = static_cast<long long>(i) * (1LL << 32) + j;
      const long long mirror = static_cast<long long>(j) * (1LL << 32) + i;
      if (q_offdiag_seen_.count(mirror) != 0) q_both_triangles_ = true;
      q_offdiag_seen_.insert(key);
    }
  }
  return true;
}

bool Reader::finalize() {
  Model& m = result_.model;
  const Int num_cols = static_cast<Int>(m.col_names.size());
  const Int num_rows = static_cast<Int>(row_type_.size());

  m.objective = objective_;
  m.constraints = SparseMatrix::from_triplets(num_rows, num_cols, std::move(a_entries_));

  m.row_lower.assign(sz(num_rows), 0.0);
  m.row_upper.assign(sz(num_rows), 0.0);
  for (Int r = 0; r < num_rows; ++r) {
    const std::size_t i = sz(r);
    const double b = row_rhs_[i];
    switch (row_type_[i]) {
      case 'E':
        m.row_lower[i] = b;
        m.row_upper[i] = b;
        break;
      case 'L':
        m.row_lower[i] = -kInf;
        m.row_upper[i] = b;
        break;
      case 'G':
        m.row_lower[i] = b;
        m.row_upper[i] = kInf;
        break;
      default:
        return fail("internal: unknown row type");
    }
    if (!row_has_range_[i]) continue;

    // RANGES, exactly as HiGHS applies it. The sign of the range value matters
    // only for equality rows; for L and G rows the magnitude is what counts.
    const double r_val = row_range_[i];
    const char ty = row_type_[i];
    if ((ty == 'E' && r_val < 0.0) || ty == 'L') {
      m.row_lower[i] = m.row_upper[i] - std::fabs(r_val);
    } else if ((ty == 'E' && r_val > 0.0) || ty == 'G') {
      m.row_upper[i] = m.row_lower[i] + std::fabs(r_val);
    }
    // An equality row with a zero range stays an equality.
  }

  if (!q_entries_.empty()) {
    std::vector<Triplet> q = std::move(q_entries_);
    if (!q_explicit_full_ && !q_both_triangles_) {
      // Only one triangle was given, so mirror the off-diagonal entries.
      std::vector<Triplet> mirrored;
      mirrored.reserve(q.size() * 2);
      for (const Triplet& e : q) {
        mirrored.push_back(e);
        if (e.row != e.col) mirrored.push_back(Triplet{e.col, e.row, e.value});
      }
      q = std::move(mirrored);
    }
    m.hessian = SparseMatrix::from_triplets(num_cols, num_cols, std::move(q));
  }

  if (negative_up_bounds_ > 0) {
    warn("negative UP bounds seen on " + std::to_string(negative_up_bounds_) +
         " column(s); reader is using the " +
         (options_.negative_up_bound == MpsOptions::NegativeUpBound::kMinusInfinity
              ? std::string("CPLEX")
              : std::string("HiGHS")) +
         " convention");
  }

  std::string error;
  if (!m.validate(&error)) return fail("model failed validation: " + error);
  result_.ok = true;
  return true;
}

MpsReadResult Reader::run(std::istream& in) {
  Section section = Section::kNone;
  std::string line;

  while (std::getline(in, line)) {
    ++line_no_;
    if (!line.empty() && line.back() == '\r') line.pop_back();  // DOS line endings
    if (line.empty()) continue;
    if (line[0] == '*') continue;

    const bool is_header = !std::isspace(static_cast<unsigned char>(line[0]));

    if (!is_header && section == Section::kRows) {
      // Buffered until the section ends and the format has been decided.
      pending_row_lines_.push_back(line);
      pending_row_line_numbers_.push_back(line_no_);
      continue;
    }

    const std::vector<std::string> t = is_header ? tokenize(line) : tokens_for(line);
    if (t.empty()) continue;

    if (is_header) {
      const Section next = section_from(t[0]);
      if (next == Section::kNone)
        return (fail("unknown section \"" + t[0] + "\""), result_);
      const int header_line = line_no_;
      if (section == Section::kRows && !flush_rows()) return result_;
      line_no_ = header_line;
      section = next;
      if (section == Section::kName && t.size() >= 2) result_.model.name = t[1];
      if (section == Section::kObjSense && t.size() >= 2) {
        const std::string w = upper(t[1]);
        result_.model.sense =
            (w == "MAX" || w == "MAXIMIZE") ? ObjSense::kMaximize : ObjSense::kMinimize;
      }
      if (section == Section::kEndata) break;
      continue;
    }

    bool ok = true;
    switch (section) {
      case Section::kObjSense: {
        const std::string w = upper(t[0]);
        result_.model.sense =
            (w == "MAX" || w == "MAXIMIZE") ? ObjSense::kMaximize : ObjSense::kMinimize;
        break;
      }
      case Section::kRows:
        ok = fail("internal: ROWS data line reached the dispatcher");
        break;
      case Section::kColumns:
        ok = handle_columns(t);
        break;
      case Section::kRhs:
        ok = handle_rhs(t);
        break;
      case Section::kRanges:
        ok = handle_ranges(t);
        break;
      case Section::kBounds:
        ok = handle_bounds(t);
        break;
      case Section::kQuadObj:
        ok = handle_quadratic(t, false);
        break;
      case Section::kQMatrix:
        ok = handle_quadratic(t, true);
        break;
      case Section::kName:
        break;  // a NAME section body is not meaningful
      case Section::kNone:
        ok = fail("data line before any section header");
        break;
      case Section::kEndata:
        break;
    }
    if (!ok) return result_;
  }

  if (!flush_rows()) return result_;
  if (!seen_objective_) {
    warn("no N row found: the objective is taken to be zero");
  }
  finalize();
  return result_;
}

}  // namespace

MpsReadResult read_mps_stream(std::istream& in, const std::string& source_label,
                              const MpsOptions& options) {
  Reader reader(options, source_label);
  return reader.run(in);
}

MpsReadResult read_mps(const std::string& path, const MpsOptions& options) {
  std::ifstream in(path);
  if (!in) {
    MpsReadResult r;
    r.ok = false;
    r.error = "cannot open \"" + path + "\"";
    return r;
  }
  return read_mps_stream(in, path, options);
}

}  // namespace sankhya
