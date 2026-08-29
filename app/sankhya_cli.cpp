#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <ios>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/mps_reader.hpp"
#include "sankhya/backend.hpp"
#include "sankhya/branch_and_bound.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/presolve.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/simplex.hpp"
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
      "  sankhya milp <file.mps> [options]     solve a MILP by branch and bound\n"
      "  sankhya qp <file.qps> [options]       solve a convex QP by ADMM\n"
      "  sankhya simplex <file.mps> [options]  solve an LP with the primal simplex\n"
      "  sankhya presolve <file.mps> [opts]    reduce a model and report what went\n"
      "                                        and what is left\n"
      "  sankhya backends                      report which backends this build has\n"
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
      "  --pdlp-termination   drop the infinity-norm criteria and stop on PDLP's\n"
      "                       2-norm ones alone\n"
      "  --no-adaptive        fixed step size\n"
      "  --no-restarts        no restarting\n"
      "  --no-halpern         averaged restarts instead of Halpern\n"
      "  --no-reflection --adaptive-step --no-fixed-point-restart\n"
      "  --no-pid-weight      turn off one cuPDLPx addition each (all on)\n"
      "  --reflection=G       R(z) = (1+G) T(z) - G z, 0 is plain Halpern\n"
      "  --constant-step      fixed step size instead of the adaptive rule\n"
      "  --step-scale=S       fixed step size S/||K|| (default 0.998)\n"
      "  --fixed-point-restart  restart on ||z-T(z)||, not on the KKT error\n"
      "  --pid-weight         PID control on the primal weight\n"
      "  --kp= --ki= --kd=    its coefficients (default 0.5, 0, 0)\n"
      "  --no-polish          no feasibility polishing\n"
      "  --no-exit-polish     polish during the solve but not on the way out\n"
      "  --gap-tol=T          duality gap tolerance, if not --tol (e.g. 1e-2)\n"
      "  --polish-first=N     first polish attempt, doubling after (default 100)\n"
      "  --polish-factor=F    polish budget as a fraction of iterations so far\n"
      "  --no-primal-weight   keep the primal weight at one\n"
      "  --presolve           reduce the model first, then map the answer back\n"
      "  --profile            report where the device time went, kernel by kernel\n"
      "  --verbose            print progress\n"
      "  --backend=cpu|cuda   force a backend instead of picking automatically\n"
      "  --solution=<path>    write the primal solution so it can be checked\n"
      "                       independently\n");
}

// "--tol=1e-6" -> 1e-6. Returns false when the argument is not this option at
// all, so a chain of these reads as a list of alternatives.
bool value_of(const std::string& arg, const std::string& prefix, double* out) {
  if (arg.rfind(prefix, 0) != 0) return false;
  *out = std::strtod(arg.c_str() + prefix.size(), nullptr);
  return true;
}

// JSON has no inf and no nan. An objective that is either means there is no
// incumbent, and "null" is what a parser can actually read.
std::string json_number(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream out;
  out.precision(17);
  out << v;
  return out.str();
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

int command_presolve(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::fprintf(stderr, "presolve: expected a file name\n");
    return 2;
  }
  bool as_json = false;
  bool quiet = false;
  sankhya::PresolveOptions options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--format=json") {
      as_json = true;
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a == "--no-bound-tightening") {
      options.bound_tightening = false;
    } else if (a == "--no-duplicate-rows") {
      options.duplicate_rows = false;
    } else if (a == "--no-forcing-rows") {
      options.forcing_rows = false;
    } else if (a == "--no-column-singletons") {
      options.free_column_singletons = false;
    } else if (a == "--no-dual-fixing") {
      options.dual_fixing = false;
    } else if (a == "--no-doubletons") {
      options.doubleton_equations = false;
    } else if (a == "--rows-only") {
      options.fixed_columns = false;
      options.empty_columns = false;
      options.free_column_singletons = false;
    } else if (a != "--format=human") {
      std::fprintf(stderr, "presolve: unknown option \"%s\"\n", a.c_str());
      return 2;
    }
  }

  const sankhya::MpsReadResult read_result = sankhya::read_mps(args[0]);
  if (!read_result.ok) {
    std::fprintf(stderr, "error: %s\n", read_result.error.c_str());
    return 1;
  }
  if (!quiet) {
    for (const std::string& w : read_result.warnings)
      std::fprintf(stderr, "warning: %s\n", w.c_str());
  }

  const sankhya::PresolveResult r = sankhya::presolve(read_result.model, options);
  const sankhya::PresolveCounts& c = r.counts;
  if (as_json) {
    std::ostringstream out;
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"status\":\"" << sankhya::to_string(r.status) << "\","
        << "\"rows_before\":" << r.original_rows << ","
        << "\"cols_before\":" << r.original_cols << ","
        << "\"nnz_before\":" << r.original_nnz << ","
        << "\"rows_after\":" << (r.original_rows - c.rows_removed) << ","
        << "\"cols_after\":" << (r.original_cols - c.cols_removed) << ","
        << "\"nnz_after\":" << (r.original_nnz - c.nonzeros_removed) << ","
        << "\"empty_rows\":" << c.empty_rows << ","
        << "\"singleton_rows\":" << c.singleton_rows << ","
        << "\"redundant_rows\":" << c.redundant_rows << ","
        << "\"forcing_rows\":" << c.forcing_rows << ","
        << "\"duplicate_rows\":" << c.duplicate_rows << ","
        << "\"fixed_columns\":" << c.fixed_columns << ","
        << "\"empty_columns\":" << c.empty_columns << ","
        << "\"free_column_singletons\":" << c.free_column_singletons << ","
        << "\"doubleton_equations\":" << c.doubleton_equations << ","
        << "\"dual_fixed_columns\":" << c.dual_fixed_columns << ","
        << "\"bounds_tightened\":" << c.bounds_tightened << ","
        << "\"rounds\":" << c.rounds << ","
        << "\"seconds\":" << r.seconds << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return 0;
  }

  std::fputs(sankhya::format_presolve(r).c_str(), stdout);
  return 0;
}

int command_simplex(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::fprintf(stderr, "simplex: expected a file name\n");
    return 2;
  }
  bool as_json = false;
  bool quiet = false;
  bool use_presolve = false;
  sankhya::SimplexOptions options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    double v = 0.0;
    if (value_of(a, "--max-iter=", &v)) {
      options.max_iterations = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--time-limit=", &v)) {
      options.time_limit_seconds = v;
    } else if (value_of(a, "--refactor=", &v)) {
      options.refactorization_frequency = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--max-rollback=", &v)) {
      options.max_rollback = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--stall=", &v)) {
      options.stall_iterations = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--primal-tol=", &v)) {
      options.primal_tolerance = v;
    } else if (value_of(a, "--dual-tol=", &v)) {
      options.dual_tolerance = v;
    } else if (a == "--dse") {
      options.dual_pricing = sankhya::SimplexOptions::DualPricing::kSteepestEdge;
    } else if (a == "--dual") {
      options.algorithm = sankhya::SimplexOptions::Algorithm::kDual;
    } else if (a == "--primal") {
      options.algorithm = sankhya::SimplexOptions::Algorithm::kPrimal;
    } else if (a == "--piecewise-phase-one") {
      options.piecewise_phase_one = true;
    } else if (a == "--dantzig") {
      options.pricing = sankhya::SimplexOptions::Pricing::kDantzig;
    } else if (a == "--devex") {
      options.pricing = sankhya::SimplexOptions::Pricing::kDevex;
    } else if (a == "--presolve") {
      use_presolve = true;
    } else if (a == "--verbose") {
      options.verbose = true;
    } else if (a == "--format=json") {
      as_json = true;
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a != "--format=human") {
      std::fprintf(stderr, "simplex: unknown option \"%s\"\n", a.c_str());
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

  sankhya::PresolveResult pre;
  sankhya::Model solved_model = read_result.model;
  if (use_presolve) {
    pre = sankhya::presolve(read_result.model);
    if (pre.status != sankhya::PresolveStatus::kReduced) {
      std::printf("status        %s (proved by presolve, without solving)\n",
                  sankhya::to_string(pre.status).c_str());
      return 1;
    }
    if (!quiet && !as_json) std::fputs(sankhya::format_presolve(pre).c_str(), stdout);
    solved_model = pre.reduced;
  }

  const sankhya::StandardFormResult sf = sankhya::to_standard_form(solved_model);
  if (!sf.ok) {
    std::fprintf(stderr, "error: %s\n", sf.error.c_str());
    return 1;
  }

  sankhya::SimplexResult r = sankhya::solve_lp(sf.lp, options);
  if (use_presolve && !r.x.empty()) r.x = pre.postsolve.apply(r.x);
  const sankhya::ModelViolation checked =
      sankhya::measure_violation(read_result.model, r.x);

  if (as_json) {
    std::ostringstream out;
    out.setf(std::ios::boolalpha);
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"status\":\"" << sankhya::to_string(r.status) << "\","
        << "\"objective\":" << r.objective << ","
        << "\"iterations\":" << r.iterations << ","
        << "\"phase_one_iterations\":" << r.phase_one_iterations << ","
        << "\"refactorizations\":" << r.refactorizations << ","
        << "\"bland_switches\":" << r.bland_switches << ","
        << "\"worst_update_growth\":" << r.worst_update_growth << ","
        << "\"rollbacks\":" << r.rollbacks << ","
        << "\"devex_resets\":" << r.devex_resets << ","
        << "\"dual_start_flips\":" << r.dual_start_flips << ","
        << "\"fell_back_to_primal\":" << r.fell_back_to_primal << ","
        << "\"seconds\":" << r.solve_seconds << ","
        << "\"presolved\":" << use_presolve << ","
        << "\"row_violation\":" << checked.row_violation << ","
        << "\"bound_violation\":" << checked.bound_violation << ","
        << "\"std_rows\":" << sf.lp.num_rows() << ","
        << "\"std_cols\":" << sf.lp.num_cols() << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return r.status == sankhya::SimplexStatus::kOptimal ? 0 : 1;
  }

  std::printf(
      "status        %s%s%s\n"
      "objective     %.12e\n"
      "iterations    %d  (%d in phase one)\n"
      "refactorized  %d times\n"
      "time          %.3f s\n"
      "vs original   row %.3e  bound %.3e\n",
      sankhya::to_string(r.status).c_str(), r.message.empty() ? "" : ": ",
      r.message.c_str(), r.objective, r.iterations, r.phase_one_iterations,
      r.refactorizations, r.solve_seconds, checked.row_violation,
      checked.bound_violation);
  return r.status == sankhya::SimplexStatus::kOptimal ? 0 : 1;
}

int command_solve(const std::vector<std::string>& args) {
  bool use_presolve = false;
  bool profile = false;
  sankhya::PresolveOptions presolve_options;
  if (args.empty()) {
    std::fprintf(stderr, "solve: expected a file name\n");
    return 2;
  }
  sankhya::PdhgOptions options;
  bool as_json = false;
  bool quiet = false;
  std::string solution_path;

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
    } else if (a == "--backend=cpu") {
      options.backend = &sankhya::cpu_backend();
    } else if (a == "--backend=cuda") {
#ifdef SANKHYA_WITH_CUDA
      options.backend = &sankhya::cuda_backend();
#else
      std::fprintf(stderr,
                   "this build has no CUDA backend; configure with "
                   "-DSANKHYA_ENABLE_CUDA=ON\n");
      return 2;
#endif
    } else if (a == "--pdlp-termination") {
      options.require_inf_norm_termination = false;
    } else if (a == "--ruiz-only") {
      options.scaling.pock_chambolle = false;
    } else if (a == "--no-scaling") {
      options.scaling.ruiz_iterations = 0;
      options.scaling.pock_chambolle = false;
    } else if (a == "--no-adaptive") {
      options.adaptive_step_size = false;
    } else if (a == "--halpern") {
      options.halpern = true;
    } else if (value_of(a, "--reflection=", &v)) {
      options.reflection = v;
    } else if (a == "--constant-step") {
      options.constant_step_size = true;
    } else if (a == "--no-reflection") {
      options.reflection = 0.0;
    } else if (a == "--adaptive-step") {
      options.constant_step_size = false;
    } else if (a == "--no-fixed-point-restart") {
      options.restart_on_fixed_point = false;
    } else if (a == "--no-pid-weight") {
      options.pid_primal_weight = false;
    } else if (value_of(a, "--step-scale=", &v)) {
      options.constant_step_size = true;
      options.constant_step_scale = v;
    } else if (a == "--fixed-point-restart") {
      options.restart_on_fixed_point = true;
    } else if (a == "--pid-weight") {
      options.pid_primal_weight = true;
    } else if (value_of(a, "--kp=", &v)) {
      options.pid_primal_weight = true;
      options.primal_weight_kp = v;
    } else if (value_of(a, "--ki=", &v)) {
      options.pid_primal_weight = true;
      options.primal_weight_ki = v;
    } else if (value_of(a, "--kd=", &v)) {
      options.pid_primal_weight = true;
      options.primal_weight_kd = v;
    } else if (a == "--cupdlpx") {
      // All four together, which is the configuration the paper reports.
      options.reflection = 1.0;
      options.constant_step_size = true;
      options.restart_on_fixed_point = true;
      options.pid_primal_weight = true;
    } else if (a == "--no-halpern") {
      options.halpern = false;
    } else if (a == "--no-polish") {
      options.polish_feasibility = false;
    } else if (a == "--no-exit-polish") {
      options.polish_on_exit = false;
    } else if (value_of(a, "--gap-tol=", &v)) {
      options.gap_tolerance = v;
    } else if (value_of(a, "--polish-first=", &v)) {
      options.polish_first_iteration = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--polish-factor=", &v)) {
      options.polish_iteration_factor = v;
    } else if (a == "--no-restarts") {
      options.restarts = false;
    } else if (a == "--no-primal-weight") {
      options.primal_weight_updates = false;
    } else if (a == "--verbose") {
      options.verbose = true;
    } else if (a == "--format=json") {
      as_json = true;
    } else if (a == "--profile") {
      profile = true;
    } else if (a == "--presolve") {
      use_presolve = true;
    } else if (a == "--presolve-no-bound-tightening") {
      use_presolve = true;
      presolve_options.bound_tightening = false;
    } else if (a == "--presolve-rows-only") {
      use_presolve = true;
      presolve_options.fixed_columns = false;
      presolve_options.empty_columns = false;
      presolve_options.free_column_singletons = false;
    } else if (a == "--presolve-no-dual-fixing") {
      use_presolve = true;
      presolve_options.dual_fixing = false;
    } else if (a == "--presolve-no-doubletons") {
      use_presolve = true;
      presolve_options.doubleton_equations = false;
    } else if (a == "--presolve-no-forcing") {
      use_presolve = true;
      presolve_options.forcing_rows = false;
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
  // Presolve sits in front of the solver and behind postsolve, so nothing
  // downstream of here knows it happened: what comes back out is a point in the
  // original model's columns, with the original model's objective.
  sankhya::PresolveResult pre;
  sankhya::Model solved_model = read_result.model;
  if (use_presolve) {
    pre = sankhya::presolve(read_result.model, presolve_options);
    if (pre.status != sankhya::PresolveStatus::kReduced) {
      std::printf("status        %s (proved by presolve, without solving)\n",
                  sankhya::to_string(pre.status).c_str());
      if (!pre.message.empty())
        std::printf("reason        %s\n", pre.message.c_str());
      return 1;
    }
    if (!quiet) std::fputs(sankhya::format_presolve(pre).c_str(), stdout);
    solved_model = pre.reduced;
  }

  const sankhya::StandardFormResult sf = sankhya::to_standard_form(solved_model);
  if (!sf.ok) {
    std::fprintf(stderr, "error: %s\n", sf.error.c_str());
    return 1;
  }

  const sankhya::LinAlgBackend& profiled_backend =
      options.backend ? *options.backend : sankhya::default_backend();
  if (profile) profiled_backend.set_profiling(true);
  sankhya::PdhgResult r = sankhya::solve_pdhg(sf.lp, options);
  std::string kernel_report;
  if (profile) {
    kernel_report = profiled_backend.profile_report();
    profiled_backend.set_profiling(false);
  }

  // The solver's residuals describe the model it was handed. With presolve in
  // front of it that is not the model the caller asked about, so the answer is
  // re-checked against the original before any claim is made about it. See
  // measure_violation in model.hpp for what went wrong without this.
  if (use_presolve) r.x = pre.postsolve.apply(r.x);

  // The dual, mapped back the same way the primal is. Standard-form rows carry
  // a sign and a model row they came from - a two-sided row becomes two of them
  // - so they fold back onto the model's own rows first, and presolve's row
  // recovery runs after that.
  std::vector<double> model_dual(sankhya::sz(solved_model.num_rows()), 0.0);
  for (sankhya::Int i = 0; i < sf.lp.num_rows(); ++i) {
    const auto& origin = sf.lp.row_origin[sankhya::sz(i)];
    model_dual[sankhya::sz(origin.model_row)] += origin.sign * r.y[sankhya::sz(i)];
  }
  std::vector<double> original_dual = model_dual;
  bool dual_exact = true;
  if (use_presolve) {
    // Reduced costs of the reduced model, which the singleton-row recovery
    // needs: d = c - A' y.
    std::vector<double> reduced_costs = solved_model.objective;
    for (sankhya::Int i = 0; i < solved_model.num_rows(); ++i) {
      const double yi = model_dual[sankhya::sz(i)];
      if (yi == 0.0) continue;
      for (sankhya::Int e = solved_model.constraints.row_begin(i);
           e < solved_model.constraints.row_end(i); ++e) {
        reduced_costs[sankhya::sz(solved_model.constraints.index()[sankhya::sz(e)])] -=
            yi * solved_model.constraints.value()[sankhya::sz(e)];
      }
    }
    original_dual = pre.postsolve.apply_dual(model_dual, reduced_costs,
                                             read_result.model.objective);
    dual_exact = pre.postsolve.dual_is_exact();
  }

  // Measured on both paths, because the number is worth seeing either way.
  const sankhya::ModelViolation checked =
      sankhya::measure_violation(read_result.model, r.x);

  // Acted on only when presolve ran, and the asymmetry is deliberate. Without
  // presolve the solver's own criteria already describe this model, and a
  // violation that survives them is PDLP's documented relative behaviour, which
  // --abs-tol exists to override. With presolve they describe the reduced model
  // instead, so nothing has checked the original unless this does.
  // Which number is checked follows which tolerance was asked for. With
  // --abs-tol the caller wants an absolute guarantee, so the absolute violation
  // is what gets tested. Without it they asked for PDLP's relative criterion,
  // and the honest comparison is the relative violation against the same --tol
  // the unpresolved path is judged by. Holding the absolute number against a
  // relative tolerance failed graph40-40 for a violation the unpresolved run
  // was passing.
  bool violates_cap = false;
  if (use_presolve && r.status == sankhya::PdhgStatus::kOptimal) {
    const bool absolute = options.absolute_tolerance > 0.0;
    const double measured =
        absolute ? checked.worst() : checked.relative_row_violation;
    const double cap =
        absolute ? options.absolute_tolerance : options.tolerance;
    if (measured > cap) {
      violates_cap = true;
      r.status = sankhya::PdhgStatus::kNumericalError;
      r.message =
          "the reduced model met the tolerance and the original does not. "
          "presolve removes rows, and the rows it removes are often the ones "
          "with the largest right-hand sides, which are what the relative "
          "criterion divides by - so the same --tol is a stricter absolute "
          "requirement on the model you handed in than on the one that was "
          "solved. re-run with --abs-tol, or a tighter --tol";
    }
  }

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
    // Row duals - shadow prices - after the solution, marked with whether
    // presolve could put them back exactly.
    out << "# duals " << (dual_exact ? "exact" : "approximate") << "\n";
    for (std::size_t i = 0; i < original_dual.size(); ++i) {
      const std::string name = i < read_result.model.row_names.size()
                                   ? read_result.model.row_names[i]
                                   : ("row" + std::to_string(i));
      out << "# dual " << name << " " << original_dual[i] << "\n";
    }
  }

  if (as_json) {
    std::ostringstream out;
    out.setf(std::ios::boolalpha);
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"status\":\"" << sankhya::to_string(r.status) << "\","
        << "\"objective\":" << r.objective << ","
        << "\"presolved\":" << use_presolve << ","
        << "\"original_row_violation\":" << checked.row_violation << ","
        << "\"original_row_violation_relative\":"
        << checked.relative_row_violation << ","
        << "\"dual_exact\":" << dual_exact << ","
        << "\"original_bound_violation\":" << checked.bound_violation << ","
        << "\"iterations\":" << r.iterations << ","
        << "\"restarts\":" << r.restarts << ","
        << "\"polish_attempts\":" << r.polish_attempts << ","
        << "\"polish_iterations\":" << r.polish_iterations << ","
        << "\"polished\":" << (r.polished ? "true" : "false") << ","
        << "\"seconds\":" << r.solve_seconds << ","
        << "\"rel_primal\":" << r.residual.relative_primal << ","
        << "\"rel_dual\":" << r.residual.relative_dual << ","
        << "\"rel_gap\":" << r.residual.relative_gap << ","
        << "\"rel_primal_inf\":" << r.residual.relative_primal_inf << ","
        << "\"rel_dual_inf\":" << r.residual.relative_dual_inf << ","
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

  char vb[220];
  std::snprintf(vb, sizeof(vb), "vs original   row %.3e  bound %.3e%s\n",
                checked.row_violation, checked.bound_violation,
                violates_cap ? "   <- over the tolerance" : "");
  const std::string violation_line(vb);
  std::printf(
      "status        %s%s%s\n"
      "objective     %.12e\n"
      "iterations    %d  (%d restarts, %d more in %d polish attempts%s)\n"
      "time          %.3f s\n"
      "relative      primal %.3e  dual %.3e  gap %.3e\n"
      "rel inf-norm  primal %.3e  dual %.3e\n"
      "worst abs     primal %.3e  dual %.3e\n"
      "||K||         %.6e\n"
      "row spread    %.3e -> %.3e\n%s",
      sankhya::to_string(r.status).c_str(), r.message.empty() ? "" : ": ",
      r.message.c_str(), r.objective, r.iterations, r.restarts,
      r.polish_iterations, r.polish_attempts, r.polished ? ", adopted" : "",
      r.solve_seconds,
      r.residual.relative_primal, r.residual.relative_dual, r.residual.relative_gap,
      r.residual.relative_primal_inf, r.residual.relative_dual_inf,
      r.residual.primal_residual_inf, r.residual.dual_residual_inf,
      r.matrix_norm_estimate, r.scaling.row_spread_before,
      r.scaling.row_spread_after, violation_line.c_str());
  if (!kernel_report.empty()) {
    std::printf(
        "\nwhere the device time went. Timing serialises the launches it\n"
        "measures, so this run is slower than a real one and the wall clock\n"
        "above is not a benchmark. The proportions are the point.\n\n%s",
        kernel_report.c_str());
  }
  return r.status == sankhya::PdhgStatus::kOptimal ? 0 : 1;
}

int command_milp(const std::vector<std::string>& args) {
  bool use_presolve = false;
  if (args.empty()) {
    std::fprintf(stderr, "milp: expected a file name\n");
    return 2;
  }
  sankhya::BranchAndBoundOptions options;
  options.relaxation.tolerance = 1e-8;
  options.relaxation.max_iterations = 500000;
  options.relaxation.time_limit_seconds = 30.0;
  bool as_json = false;

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    double v = 0.0;
    if (value_of(a, "--node-limit=", &v)) {
      options.node_limit = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--time-limit=", &v)) {
      options.time_limit_seconds = v;
    } else if (value_of(a, "--lp-tol=", &v)) {
      options.relaxation.tolerance = v;
    } else if (value_of(a, "--lp-max-iter=", &v)) {
      options.relaxation.max_iterations = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--lp-time=", &v)) {
      options.relaxation.time_limit_seconds = v;
    } else if (value_of(a, "--int-tol=", &v)) {
      options.integrality_tolerance = v;
    } else if (value_of(a, "--gomory-rounds=", &v)) {
      options.gomory_rounds = static_cast<sankhya::Int>(v);
    } else if (a == "--gomory") {
      options.gomory_cuts = true;
    } else if (a == "--no-reduced-cost-fixing") {
      options.reduced_cost_fixing = false;
    } else if (a == "--no-adaptive-cuts") {
      options.adaptive_cuts = false;
    } else if (value_of(a, "--cut-threshold=", &v)) {
      options.cut_bound_improvement = v;
    } else if (a == "--no-gomory") {
      options.gomory_cuts = false;
    } else if (a == "--no-cuts") {
      options.root_cuts = false;
    } else if (value_of(a, "--stall=", &v)) {
      options.simplex.dual_stall_iterations = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--dive-iters=", &v)) {
      options.dive_iteration_factor = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--node-iters=", &v)) {
      options.node_iteration_factor = static_cast<sankhya::Int>(v);
    } else if (a == "--dse") {
      options.simplex.dual_pricing =
          sankhya::SimplexOptions::DualPricing::kSteepestEdge;
    } else if (a == "--nodes=simplex") {
      options.node_solver = sankhya::BranchAndBoundOptions::NodeSolver::kSimplex;
    } else if (a == "--nodes=first-order") {
      options.node_solver =
          sankhya::BranchAndBoundOptions::NodeSolver::kFirstOrder;
    } else if (a == "--most-fractional") {
      options.branching =
          sankhya::BranchAndBoundOptions::Branching::kMostFractional;
    } else if (a == "--no-heuristic") {
      options.rounding_heuristic = false;
      options.diving_heuristic = false;
    } else if (a == "--presolve") {
      use_presolve = true;
    } else if (a == "--verbose") {
      options.verbose = true;
    } else if (a == "--format=json") {
      as_json = true;
    } else if (a != "--quiet" && a != "--format=human") {
      std::fprintf(stderr, "milp: unknown option \"%s\"\n", a.c_str());
      return 2;
    }
  }

  const sankhya::MpsReadResult read_result = sankhya::read_mps(args[0]);
  if (!read_result.ok) {
    std::fprintf(stderr, "error: %s\n", read_result.error.c_str());
    return 1;
  }
  if (!read_result.model.has_integers()) {
    std::fprintf(stderr, "warning: no integer variables; this is just an LP\n");
  }

  const bool quiet = as_json;
  sankhya::PresolveResult pre;
  sankhya::Model solved_model = read_result.model;
  if (use_presolve) {
    pre = sankhya::presolve(read_result.model);
    if (pre.status != sankhya::PresolveStatus::kReduced) {
      std::printf("status        %s (proved by presolve, without branching)\n",
                  sankhya::to_string(pre.status).c_str());
      if (!pre.message.empty())
        std::printf("reason        %s\n", pre.message.c_str());
      return 1;
    }
    if (!quiet) std::fputs(sankhya::format_presolve(pre).c_str(), stdout);
    solved_model = pre.reduced;
  }

  sankhya::BranchAndBoundResult r = sankhya::solve_milp(solved_model, options);
  if (use_presolve) r.x = pre.postsolve.apply(r.x);

  if (as_json) {
    std::ostringstream out;
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"status\":\"" << sankhya::to_string(r.status) << "\","
        << "\"objective\":" << json_number(r.objective) << ","
        << "\"dual_bound\":" << json_number(r.dual_bound) << ","
        << "\"gap\":" << r.relative_gap << ","
        << "\"nodes\":" << r.nodes << ","
        << "\"max_depth\":" << r.max_depth << ","
        << "\"relaxations\":" << r.relaxations_solved << ","
        << "\"incumbents\":" << r.incumbents_found << ","
        << "\"heuristic_successes\":" << r.heuristic_successes << ","
        << "\"cuts_added\":" << r.cuts_added << ","
        << "\"gomory_cuts_added\":" << r.gomory_cuts_added << ","
        << "\"cuts_discarded\":" << (r.cuts_discarded ? "true" : "false") << ","
        << "\"root_bound_rise\":" << r.root_bound_rise << ","
        << "\"warm_started_nodes\":" << r.warm_started_nodes << ","
        << "\"simplex_iterations\":" << r.simplex_iterations << ","
        << "\"reduced_cost_tightenings\":" << r.reduced_cost_tightenings << ","
        << "\"reduced_cost_fixings\":" << r.reduced_cost_fixings << ","
        << "\"root_before\":" << r.root_bound_before_cuts << ","
        << "\"root_after\":" << r.root_bound_after_cuts << ","
        << "\"seconds\":" << r.solve_seconds << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return r.status == sankhya::MilpStatus::kOptimal ? 0 : 1;
  }

  std::printf(
      "status        %s\n"
      "objective     %.12e\n"
      "dual bound    %.12e   (gap %.3e)\n"
      "nodes         %d   max depth %d   relaxations %d   incumbents %d\n"
      "pruned        %d proved infeasible   %d unconverged\n"
      "root cuts     %d added   bound %.10e -> %.10e\n"
      "time          %.3f s\n",
      sankhya::to_string(r.status).c_str(), r.objective, r.dual_bound,
      r.relative_gap, r.nodes, r.max_depth, r.relaxations_solved,
      r.incumbents_found, r.nodes_proved_infeasible, r.nodes_relaxation_failed,
      r.cuts_added, r.root_bound_before_cuts, r.root_bound_after_cuts,
      r.solve_seconds);
  if (!r.message.empty()) std::printf("note          %s\n", r.message.c_str());
  return r.status == sankhya::MilpStatus::kOptimal ? 0 : 1;
}

int command_qp(const std::vector<std::string>& args) {
  bool use_presolve = false;
  if (args.empty()) {
    std::fprintf(stderr, "qp: expected a file name\n");
    return 2;
  }
  sankhya::QpOptions options;
  bool as_json = false;

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    double v = 0.0;
    if (value_of(a, "--tol=", &v)) {
      options.absolute_tolerance = v;
      options.relative_tolerance = v;
    } else if (value_of(a, "--max-iter=", &v)) {
      options.max_iterations = static_cast<sankhya::Int>(v);
    } else if (value_of(a, "--time-limit=", &v)) {
      options.time_limit_seconds = v;
    } else if (value_of(a, "--abs-cap=", &v)) {
      options.max_absolute_residual = v;
    } else if (value_of(a, "--rho=", &v)) {
      options.rho = v;
    } else if (value_of(a, "--cg-iter=", &v)) {
      options.cg_max_iterations = static_cast<sankhya::Int>(v);
    } else if (a == "--no-scaling") {
      options.scaling = false;
    } else if (a == "--osqp-termination") {
      options.max_absolute_residual = 0.0;
    } else if (a == "--presolve") {
      use_presolve = true;
    } else if (value_of(a, "--max-fill=", &v)) {
      options.max_fill_ratio = v;
    } else if (a == "--indirect") {
      options.direct = false;
    } else if (a == "--direct") {
      options.direct = true;
    } else if (a == "--polish") {
      options.polish = true;
    } else if (a == "--no-adaptive-rho") {
      options.adaptive_rho = false;
    } else if (a == "--presolve") {
      use_presolve = true;
    } else if (a == "--verbose") {
      options.verbose = true;
    } else if (a == "--format=json") {
      as_json = true;
    } else if (a != "--quiet" && a != "--format=human") {
      std::fprintf(stderr, "qp: unknown option \"%s\"\n", a.c_str());
      return 2;
    }
  }

  const sankhya::MpsReadResult read_result = sankhya::read_mps(args[0]);
  if (!read_result.ok) {
    std::fprintf(stderr, "error: %s\n", read_result.error.c_str());
    return 1;
  }
  // A Hessian switches off every column-removing reduction, so what a QP gets
  // out of this is the row side: rows that cannot bind, rows that are really
  // bounds, and tighter bounds. Q carries over untouched.
  sankhya::PresolveResult pre;
  sankhya::Model solved_model = read_result.model;
  if (use_presolve) {
    pre = sankhya::presolve(read_result.model);
    if (pre.status != sankhya::PresolveStatus::kReduced) {
      std::printf("status        %s (proved by presolve, without solving)\n",
                  sankhya::to_string(pre.status).c_str());
      if (!pre.message.empty())
        std::printf("reason        %s\n", pre.message.c_str());
      return 1;
    }
    if (!as_json) std::fputs(sankhya::format_presolve(pre).c_str(), stdout);
    solved_model = pre.reduced;
  }

  sankhya::QpResult r = sankhya::solve_qp(solved_model, options);
  if (use_presolve) r.x = pre.postsolve.apply(r.x);

  if (as_json) {
    std::ostringstream out;
    out.precision(17);
    out << "{\"name\":\"" << read_result.model.name << "\","
        << "\"status\":\"" << sankhya::to_string(r.status) << "\","
        << "\"objective\":" << r.objective << ","
        << "\"iterations\":" << r.iterations << ","
        << "\"cg_iterations\":" << r.cg_iterations << ","
        << "\"rho_updates\":" << r.rho_updates << ","
        << "\"kkt_fill_ratio\":" << r.kkt_fill_ratio << ","
        << "\"fell_back_to_cg\":" << r.fell_back_to_cg << ","
        << "\"polished\":" << (r.polished ? 1 : 0) << ","
        << "\"primal\":" << r.residual.primal << ","
        << "\"dual\":" << r.residual.dual << ","
        << "\"seconds\":" << r.solve_seconds << "}\n";
    std::fputs(out.str().c_str(), stdout);
    return r.status == sankhya::QpStatus::kOptimal ? 0 : 1;
  }

  std::printf(
      "status        %s%s%s\n"
      "objective     %.12e\n"
      "iterations    %d ADMM, %d conjugate gradient   (%d rho updates)\n"
      "polish        %s\n"
      "residual      primal %.3e / %.3e   dual %.3e / %.3e\n"
      "time          %.3f s\n",
      sankhya::to_string(r.status).c_str(), r.message.empty() ? "" : ": ",
      r.message.c_str(), r.objective, r.iterations, r.cg_iterations, r.rho_updates,
      r.polished ? "accepted" : "not accepted", r.residual.primal,
      r.residual.primal_tolerance, r.residual.dual, r.residual.dual_tolerance,
      r.solve_seconds);
  return r.status == sankhya::QpStatus::kOptimal ? 0 : 1;
}

int command_backends() {
  std::printf("default       %s\n", sankhya::default_backend().name().c_str());
  std::printf("cpu           available\n");
#ifdef SANKHYA_WITH_CUDA
  bool cuda_ok = true;
  std::string why;
  try {
    (void)sankhya::cuda_backend();
  } catch (const std::exception& e) {
    cuda_ok = false;
    why = e.what();
  }
  std::printf("cuda          %s%s%s\n", cuda_ok ? "available" : "built, unusable",
              cuda_ok ? "" : ": ", why.c_str());
#else
  std::printf("cuda          not built (configure with -DSANKHYA_ENABLE_CUDA=ON)\n");
#endif
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
  if (command == "standard") return command_standard(rest);
  if (command == "presolve") return command_presolve(rest);
  if (command == "simplex") return command_simplex(rest);
  if (command == "solve") return command_solve(rest);
  if (command == "milp") return command_milp(rest);
  if (command == "qp") return command_qp(rest);
  if (command == "backends") return command_backends();

  std::fprintf(stderr, "unknown command \"%s\"\n", command.c_str());
  print_usage();
  return 2;
}
