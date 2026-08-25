#include <cstdio>
#include <cstdlib>
#include <ios>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/standard_form.hpp"

namespace {

void print_usage() {
  std::printf(
      "sankhya - optimization solver core\n"
      "\n"
      "usage:\n"
      "  sankhya read <file.mps> [options]     read a model and print its statistics\n"
      "  sankhya standard <file.mps> [options] read a model and build the solver's\n"
      "                                        standard form, printing its shape\n"
      "  sankhya solve <file.mps> [options]    solve an LP with the first-order method\n"
      "\n"
      "options:\n"
      "  --neg-up-bound=keep|minus-inf   how to read a negative UP bound with no\n"
      "                                  lower bound. keep (default) matches HiGHS,\n"
      "                                  minus-inf matches CPLEX.\n"
      "  --quiet                         do not print reader warnings\n"
      "\n"
      "solve options:\n"
      "  --tol=<x>            relative tolerance, default 1e-6\n"
      "  --abs-tol=<x>        also require no single row violated by more than\n"
      "                       this in absolute terms. 0 (default) is PDLP\n"
      "                       behaviour; the relative measure alone can hide a\n"
      "                       real violation on models with a large right-hand side\n"
      "  --max-iter=<n>       iteration limit\n"
      "  --time-limit=<s>     wall clock limit in seconds\n"
      "  --no-scaling         turn off Ruiz and Pock-Chambolle preconditioning\n"
      "  --ruiz-only          Ruiz equilibration only, no Pock-Chambolle pass\n"
      "  --no-adaptive        fixed step size\n"
      "  --no-restarts        no restarting\n"
      "  --no-primal-weight   keep the primal weight at one\n"
      "  --verbose            print progress\n"
      "  --solution=<path>    write the primal solution so it can be checked\n"
      "                       independently\n");
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

int command_standard(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::fprintf(stderr, "standard: expected a file name\n");
    return 2;
  }
  bool as_json = false;
  bool quiet = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--format=json") {
      as_json = true;
    } else if (args[i] == "--quiet") {
      quiet = true;
    } else if (args[i] != "--format=human") {
      std::fprintf(stderr, "standard: unknown option \"%s\"\n", args[i].c_str());
      return 2;
    }
  }

  const sankhya::MpsReadResult read_result = sankhya::read_mps(args[0]);
  if (!read_result.ok) {
    std::fprintf(stderr, "error: %s\n", read_result.error.c_str());
    return 1;
  }
  const sankhya::StandardFormResult sf = sankhya::to_standard_form(read_result.model);
  if (!quiet) {
    for (const std::string& w : sf.warnings)
      std::fprintf(stderr, "warning: %s\n", w.c_str());
  }
  if (!sf.ok) {
    std::fprintf(stderr, "error: %s\n", sf.error.c_str());
    return 1;
  }

  const sankhya::StandardLp& lp = sf.lp;
  if (as_json) {
    std::ostringstream out;
    out.setf(std::ios::boolalpha);
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"model_rows\":" << read_result.model.num_rows() << ","
        << "\"model_nnz\":" << read_result.model.constraints.nnz() << ","
        << "\"std_rows\":" << lp.num_rows() << ","
        << "\"std_cols\":" << lp.num_cols() << ","
        << "\"std_nnz\":" << lp.k.nnz() << ","
        << "\"equalities\":" << lp.num_equalities << ","
        << "\"inequalities\":" << lp.num_inequalities() << ","
        << "\"from_ranges\":" << sf.rows_from_ranges << ","
        << "\"free_rows_dropped\":" << sf.free_rows_dropped << ","
        << "\"maximize\":" << (lp.objective_scale < 0.0) << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return 0;
  }

  std::printf(
      "model         %d rows, %d cols, %d nonzeros\n"
      "standard form %d rows, %d cols, %d nonzeros\n"
      "  equalities  %d\n"
      "  >= rows     %d  (%d of them from %d two-sided rows)\n"
      "  dropped     %d free rows\n"
      "objective     %s, offset %.10e\n",
      read_result.model.num_rows(), read_result.model.num_cols(),
      read_result.model.constraints.nnz(), lp.num_rows(), lp.num_cols(), lp.k.nnz(),
      lp.num_equalities, lp.num_inequalities(), 2 * sf.rows_from_ranges,
      sf.rows_from_ranges, sf.free_rows_dropped,
      lp.objective_scale < 0.0 ? "maximize (negated)" : "minimize",
      lp.objective_offset);
  return 0;
}

int command_solve(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::fprintf(stderr, "solve: expected a file name\n");
    return 2;
  }
  sankhya::PdhgOptions options;
  bool as_json = false;
  bool quiet = false;
  std::string solution_path;

  auto value_of = [](const std::string& arg, const std::string& prefix, double* out) {
    if (arg.rfind(prefix, 0) != 0) return false;
    *out = std::strtod(arg.c_str() + prefix.size(), nullptr);
    return true;
  };

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    double v = 0.0;
    if (value_of(a, "--tol=", &v)) {
      options.tolerance = v;
    } else if (value_of(a, "--abs-tol=", &v)) {
      options.absolute_tolerance = v;
    } else if (value_of(a, "--max-iter=", &v)) {
      options.max_iterations = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--time-limit=", &v)) {
      options.time_limit_seconds = v;
    } else if (value_of(a, "--check-every=", &v)) {
      options.termination_check_frequency = static_cast<sankhya::Int>(v);
    } else if (a.rfind("--solution=", 0) == 0) {
      solution_path = a.substr(std::string("--solution=").size());
    } else if (a == "--ruiz-only") {
      options.scaling.pock_chambolle = false;
    } else if (a == "--no-scaling") {
      options.scaling.ruiz_iterations = 0;
      options.scaling.pock_chambolle = false;
    } else if (a == "--no-adaptive") {
      options.adaptive_step_size = false;
    } else if (a == "--no-restarts") {
      options.restarts = false;
    } else if (a == "--no-primal-weight") {
      options.primal_weight_updates = false;
    } else if (a == "--verbose") {
      options.verbose = true;
    } else if (a == "--format=json") {
      as_json = true;
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a != "--format=human") {
      std::fprintf(stderr, "solve: unknown option \"%s\"\n", a.c_str());
      return 2;
    }
  }

  const sankhya::MpsReadResult read_result = sankhya::read_mps(args[0]);
  if (!read_result.ok) {
    std::fprintf(stderr, "error: %s\n", read_result.error.c_str());
    return 1;
  }
  if (read_result.model.has_integers() && !quiet) {
    std::fprintf(stderr,
                 "warning: model has integer variables; solving the continuous "
                 "relaxation\n");
  }
  const sankhya::StandardFormResult sf = sankhya::to_standard_form(read_result.model);
  if (!sf.ok) {
    std::fprintf(stderr, "error: %s\n", sf.error.c_str());
    return 1;
  }

  const sankhya::PdhgResult r = sankhya::solve_pdhg(sf.lp, options);

  if (!solution_path.empty()) {
    // Column name and value, one per line. Deliberately plain text: the point is
    // that something which shares no code with this program can read it back and
    // check the answer for itself.
    std::ofstream out(solution_path);
    if (!out) {
      std::fprintf(stderr, "cannot write solution to \"%s\"\n",
                   solution_path.c_str());
      return 1;
    }
    out.precision(17);
    out << "# objective " << r.objective << "\n";
    out << "# status " << sankhya::to_string(r.status) << "\n";
    for (std::size_t j = 0; j < r.x.size(); ++j) {
      out << read_result.model.col_names[j] << " " << r.x[j] << "\n";
    }
  }

  if (as_json) {
    std::ostringstream out;
    out.setf(std::ios::boolalpha);
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"status\":\"" << sankhya::to_string(r.status) << "\","
        << "\"objective\":" << r.objective << ","
        << "\"iterations\":" << r.iterations << ","
        << "\"restarts\":" << r.restarts << ","
        << "\"seconds\":" << r.solve_seconds << ","
        << "\"rel_primal\":" << r.residual.relative_primal << ","
        << "\"rel_dual\":" << r.residual.relative_dual << ","
        << "\"rel_gap\":" << r.residual.relative_gap << ","
        << "\"abs_primal\":" << r.residual.primal_residual_inf << ","
        << "\"abs_dual\":" << r.residual.dual_residual_inf << ","
        << "\"matrix_norm\":" << r.matrix_norm_estimate << ","
        << "\"row_spread_before\":" << r.scaling.row_spread_before << ","
        << "\"row_spread_after\":" << r.scaling.row_spread_after << ","
        << "\"std_rows\":" << sf.lp.num_rows() << ","
        << "\"std_cols\":" << sf.lp.num_cols() << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return r.status == sankhya::PdhgStatus::kOptimal ? 0 : 1;
  }

  std::printf(
      "status        %s%s%s\n"
      "objective     %.12e\n"
      "iterations    %d  (%d restarts)\n"
      "time          %.3f s\n"
      "relative      primal %.3e  dual %.3e  gap %.3e\n"
      "worst abs     primal %.3e  dual %.3e\n"
      "||K||         %.6e\n"
      "row spread    %.3e -> %.3e\n",
      sankhya::to_string(r.status).c_str(), r.message.empty() ? "" : ": ",
      r.message.c_str(), r.objective, r.iterations, r.restarts, r.solve_seconds,
      r.residual.relative_primal, r.residual.relative_dual, r.residual.relative_gap,
      r.residual.primal_residual_inf, r.residual.dual_residual_inf,
      r.matrix_norm_estimate, r.scaling.row_spread_before, r.scaling.row_spread_after);
  return r.status == sankhya::PdhgStatus::kOptimal ? 0 : 1;
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
  if (command == "standard") return command_standard(rest);
  if (command == "solve") return command_solve(rest);

  std::fprintf(stderr, "unknown command \"%s\"\n", command.c_str());
  print_usage();
  return 2;
}
