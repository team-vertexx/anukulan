#include <cstdio>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/mps_reader.hpp"

namespace {

void print_usage() {
  std::printf(
      "sankhya - optimization solver core\n"
      "\n"
      "usage:\n"
      "  sankhya read <file.mps> [options]     read a model and print its statistics\n"
      "\n"
      "options:\n"
      "  --neg-up-bound=keep|minus-inf   how to read a negative UP bound with no\n"
      "                                  lower bound. keep (default) matches HiGHS,\n"
      "                                  minus-inf matches CPLEX.\n"
      "  --quiet                         do not print reader warnings\n");
}

int command_read(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::fprintf(stderr, "read: expected a file name\n");
    return 2;
  }
  const std::string path = args[0];
  sankhya::MpsOptions options;
  bool quiet = false;
  bool as_json = false;

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--neg-up-bound=keep") {
      options.negative_up_bound = sankhya::MpsOptions::NegativeUpBound::kKeepLower;
    } else if (a == "--neg-up-bound=minus-inf") {
      options.negative_up_bound = sankhya::MpsOptions::NegativeUpBound::kMinusInfinity;
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a == "--format=json") {
      as_json = true;
    } else if (a == "--format=human") {
      as_json = false;
    } else {
      std::fprintf(stderr, "read: unknown option \"%s\"\n", a.c_str());
      return 2;
    }
  }

  const sankhya::MpsReadResult result = sankhya::read_mps(path, options);
  if (!quiet) {
    for (const std::string& w : result.warnings)
      std::fprintf(stderr, "warning: %s\n", w.c_str());
  }
  if (!result.ok) {
    std::fprintf(stderr, "error: %s\n", result.error.c_str());
    return 1;
  }

  const sankhya::ModelStats stats = sankhya::compute_stats(result.model);
  if (!as_json) {
    std::printf("%s", sankhya::format_stats(result.model, stats).c_str());
    std::printf("obj nonzeros  %d\nfree rows     %d dropped\n",
                result.objective_nonzeros, result.free_rows_dropped);
    // The Netlib index counts the objective row and its entries, so print the
    // comparable figures too - it makes checking against that table trivial.
    std::printf("netlib-style  rows %d, cols %d, nonzeros %d\n", stats.rows + 1,
                stats.cols, stats.nnz + result.objective_nonzeros);
    return 0;
  }

  // Built with a stream rather than printf: a format/argument mismatch here is
  // silent (it produced a garbage warning count once), and this JSON is what the
  // benchmark harness consumes, so the types have to be checked by the compiler.
  std::ostringstream out;
  out.setf(std::ios::boolalpha);
  out.precision(17);

  std::string name;
  for (const char c : result.model.name) {
    if (c == '"' || c == '\\') name.push_back('\\');
    name.push_back(c);
  }

  auto field = [&out](const char* key, auto value, bool last = false) {
    out << '"' << key << "\":" << value << (last ? "" : ",");
  };

  out << "{\"name\":\"" << name << "\",";
  field("rows", stats.rows);
  field("cols", stats.cols);
  field("nnz", stats.nnz);
  field("obj_nnz", result.objective_nonzeros);
  field("free_rows_dropped", result.free_rows_dropped);
  field("hessian_nnz", stats.hessian_nnz);
  // The Netlib index counts the objective row and its entries, so emit the
  // comparable figures too and let the harness compare like with like.
  field("netlib_rows", stats.rows + 1);
  field("netlib_cols", stats.cols);
  field("netlib_nnz", stats.nnz + result.objective_nonzeros);
  field("eq_rows", stats.equality_rows);
  field("le_rows", stats.less_rows);
  field("ge_rows", stats.greater_rows);
  field("range_rows", stats.range_rows);
  field("free_rows", stats.free_rows);
  field("integer_cols", stats.integer_cols);
  field("binary_cols", stats.binary_cols);
  field("free_cols", stats.free_cols);
  field("boxed_cols", stats.boxed_cols);
  field("fixed_cols", stats.fixed_cols);
  field("min_abs_coeff", stats.min_abs_coeff);
  field("max_abs_coeff", stats.max_abs_coeff);
  field("obj_offset", result.model.objective_offset);
  field("maximize", result.model.sense == sankhya::ObjSense::kMaximize);
  field("fixed_format", result.fixed_format);
  field("warnings", result.warnings.size(), /*last=*/true);
  out << "}\n";
  std::fputs(out.str().c_str(), stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> argv_all(argv + 1, argv + argc);
  if (argv_all.empty() || argv_all[0] == "-h" || argv_all[0] == "--help") {
    print_usage();
    return argv_all.empty() ? 2 : 0;
  }
  const std::string command = argv_all[0];
  const std::vector<std::string> rest(argv_all.begin() + 1, argv_all.end());
  if (command == "read") return command_read(rest);

  std::fprintf(stderr, "unknown command \"%s\"\n", command.c_str());
  print_usage();
  return 2;
}
