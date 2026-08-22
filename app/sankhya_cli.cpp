#include <cstdio>
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

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--neg-up-bound=keep") {
      options.negative_up_bound = sankhya::MpsOptions::NegativeUpBound::kKeepLower;
    } else if (a == "--neg-up-bound=minus-inf") {
      options.negative_up_bound = sankhya::MpsOptions::NegativeUpBound::kMinusInfinity;
    } else if (a == "--quiet") {
      quiet = true;
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
  std::printf("%s", sankhya::format_stats(result.model, stats).c_str());
  std::printf("obj nonzeros  %d\nfree rows     %d dropped\n", result.objective_nonzeros,
              result.free_rows_dropped);
  // The Netlib index counts the objective row and its entries, so print the
  // comparable figures too - it makes checking against that table trivial.
  std::printf("netlib-style  rows %d, cols %d, nonzeros %d\n", stats.rows + 1,
              stats.cols, stats.nnz + result.objective_nonzeros);
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
