#include "anukulan/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace anukulan {
namespace {

// The symmetric adjacency of a pattern, without the diagonal and without
// duplicates, as a CSR pair. Either triangle or both may be given.
void symmetric_adjacency(const SparseMatrix& pattern, std::vector<Int>* start,
                         std::vector<Int>* index) {
  const Int n = pattern.rows();
  std::vector<Int> count(sz(n) + 1, 0);
  for (Int i = 0; i < n; ++i) {
    for (Int e = pattern.row_begin(i); e < pattern.row_end(i); ++e) {
      const Int j = pattern.index()[sz(e)];
      if (j == i || j < 0 || j >= n) continue;
      ++count[sz(i) + 1];
      ++count[sz(j) + 1];
    }
  }
  for (Int i = 0; i < n; ++i) count[sz(i) + 1] += count[sz(i)];
  std::vector<Int> raw(sz(count[sz(n)]));
  std::vector<Int> cursor(count.begin(), count.end() - 1);
  for (Int i = 0; i < n; ++i) {
    for (Int e = pattern.row_begin(i); e < pattern.row_end(i); ++e) {
      const Int j = pattern.index()[sz(e)];
      if (j == i || j < 0 || j >= n) continue;
      raw[sz(cursor[sz(i)]++)] = j;
      raw[sz(cursor[sz(j)]++)] = i;
    }
  }
  // Both triangles given means every edge arrived twice.
  start->assign(sz(n) + 1, 0);
  index->clear();
  index->reserve(raw.size());
  std::vector<Int> seen(sz(n), -1);
  for (Int i = 0; i < n; ++i) {
    for (Int p = count[sz(i)]; p < count[sz(i) + 1]; ++p) {
      const Int j = raw[sz(p)];
      if (seen[sz(j)] == i) continue;
      seen[sz(j)] = i;
      index->push_back(j);
    }
    (*start)[sz(i) + 1] = static_cast<Int>(index->size());
  }
}

// The state of a node of the quotient graph.
enum class Node : unsigned char {
  kVariable,  // not yet eliminated, and the principal of its supervariable
  kElement,   // eliminated; stands for the clique it created
  kAbsorbed,  // an element swallowed by a newer one
  kMerged,    // a variable indistinguishable from another, now part of it
  kGone,      // eliminated alongside a pivot (mass elimination)
  kDense,     // set aside and ordered last
};

}  // namespace

// Approximate minimum degree on the quotient graph.
//
// Eliminating a node of a graph makes its neighbours a clique, and the next
// pivot should be the node with the fewest neighbours in that updated graph.
// Doing that literally costs the fill itself in memory and more in time,
// because every clique is written out edge by edge. The quotient graph writes
// each clique once, as an "element" that lists its members, so a variable's
// neighbourhood is its remaining original edges plus the elements it belongs
// to. Four ideas from Amestoy, Davis and Duff make it fast:
//
//   supervariables    variables with identical neighbourhoods are merged and
//                     from then on eliminated as one, with a weight
//   mass elimination  a variable whose whole neighbourhood is inside the new
//                     element is eliminated with the pivot, for free
//   element absorption  an element whose members all joined the new element
//                     carries no information any more and is dropped, both
//                     when the pivot touched it and when a scan finds it empty
//   approximate degree  the exact degree needs the size of a union of
//                     elements; the bound
//                        |A_i| + |L_p \ i| + sum over e != p of |L_e \ L_p|
//                     needs only the set differences, which one scan over the
//                     new element's members produces for every element at once
//
// Written for this code from the published algorithm. The lists are ordinary
// vectors per node rather than one workspace with garbage collection, which
// costs some allocation and buys code that can be read.
std::vector<Int> approximate_minimum_degree(const SparseMatrix& pattern,
                                            const CholeskyOptions& options) {
  const Int n = pattern.rows();
  std::vector<Int> order;
  order.reserve(sz(n));
  if (n == 0) return order;

  std::vector<Int> adj_start;
  std::vector<Int> adj_index;
  symmetric_adjacency(pattern, &adj_start, &adj_index);

  std::vector<Node> state(sz(n), Node::kVariable);
  const double dense_limit =
      std::fmax(static_cast<double>(options.dense_row_minimum),
                options.dense_row_factor * std::sqrt(static_cast<double>(n)));
  std::vector<Int> dense;
  for (Int i = 0; i < n; ++i) {
    const Int degree = adj_start[sz(i) + 1] - adj_start[sz(i)];
    if (static_cast<double>(degree) > dense_limit && n > 2) {
      state[sz(i)] = Node::kDense;
      dense.push_back(i);
    }
  }

  std::vector<std::vector<Int>> vars(sz(n));     // A_i: variable neighbours
  std::vector<std::vector<Int>> elems(sz(n));    // E_i: element neighbours
  std::vector<std::vector<Int>> members(sz(n));  // L_e: an element's variables
  std::vector<std::vector<Int>> group(sz(n));    // variables merged into i
  std::vector<Int> weight(sz(n), 1);
  std::vector<Int> degree(sz(n), 0);
  std::vector<Int> element_size(sz(n), 0);
  for (Int i = 0; i < n; ++i) {
    if (state[sz(i)] != Node::kVariable) continue;
    for (Int p = adj_start[sz(i)]; p < adj_start[sz(i) + 1]; ++p) {
      const Int j = adj_index[sz(p)];
      if (state[sz(j)] == Node::kVariable) vars[sz(i)].push_back(j);
    }
    degree[sz(i)] = static_cast<Int>(vars[sz(i)].size());
  }
  adj_start.clear();
  adj_index.clear();

  // Degree lists: a doubly linked list per degree.
  std::vector<Int> head(sz(n) + 1, -1);
  std::vector<Int> next(sz(n), -1);
  std::vector<Int> prev(sz(n), -1);
  auto insert = [&](Int i) {
    const Int d = degree[sz(i)];
    next[sz(i)] = head[sz(d)];
    prev[sz(i)] = -1;
    if (head[sz(d)] >= 0) prev[sz(head[sz(d)])] = i;
    head[sz(d)] = i;
  };
  auto remove = [&](Int i) {
    if (prev[sz(i)] >= 0) {
      next[sz(prev[sz(i)])] = next[sz(i)];
    } else {
      head[sz(degree[sz(i)])] = next[sz(i)];
    }
    if (next[sz(i)] >= 0) prev[sz(next[sz(i)])] = prev[sz(i)];
    next[sz(i)] = -1;
    prev[sz(i)] = -1;
  };
  Int live = 0;
  for (Int i = 0; i < n; ++i) {
    if (state[sz(i)] == Node::kVariable) {
      insert(i);
      ++live;
    }
  }

  // Stamps rather than clearing: flag[i] == stamp means "in the new element",
  // difference_stamp[e] == stamp means difference[e] was set in this step.
  std::vector<Int> flag(sz(n), -1);
  std::vector<Int> difference_stamp(sz(n), -1);
  std::vector<Int> difference(sz(n), 0);
  std::vector<Int> seen(sz(n), -1);
  Int stamp = 0;
  Int seen_stamp = 0;

  auto emit = [&](Int i) {
    order.push_back(i);
    for (const Int j : group[sz(i)]) order.push_back(j);
    std::vector<Int>().swap(group[sz(i)]);
  };
  auto release = [](std::vector<Int>* v) { std::vector<Int>().swap(*v); };

  Int eliminated = 0;
  Int min_degree = 0;
  std::vector<Int> new_element;
  std::vector<std::pair<std::size_t, Int>> hashes;
  while (eliminated < live) {
    while (min_degree <= n && head[sz(min_degree)] < 0) ++min_degree;
    if (min_degree > n) break;  // cannot happen while variables remain
    const Int pivot = head[sz(min_degree)];
    remove(pivot);
    ++stamp;

    // The new element: everything the pivot reaches, directly or through an
    // element it belongs to. Those elements are inside the new one now.
    new_element.clear();
    Int element_weight = 0;
    flag[sz(pivot)] = stamp;
    auto take = [&](Int i) {
      if (state[sz(i)] != Node::kVariable || flag[sz(i)] == stamp) return;
      flag[sz(i)] = stamp;
      new_element.push_back(i);
      element_weight += weight[sz(i)];
    };
    for (const Int e : elems[sz(pivot)]) {
      if (state[sz(e)] != Node::kElement) continue;
      for (const Int i : members[sz(e)]) take(i);
      state[sz(e)] = Node::kAbsorbed;
      release(&members[sz(e)]);
    }
    for (const Int i : vars[sz(pivot)]) take(i);
    release(&vars[sz(pivot)]);
    release(&elems[sz(pivot)]);
    state[sz(pivot)] = Node::kElement;
    emit(pivot);
    eliminated += weight[sz(pivot)];
    for (const Int i : new_element) remove(i);

    // |L_e \ L_p| for every element that shares a variable with the new one,
    // in one pass: start at |L_e| and subtract each shared variable's weight.
    for (const Int i : new_element) {
      for (const Int e : elems[sz(i)]) {
        if (state[sz(e)] != Node::kElement) continue;
        if (difference_stamp[sz(e)] != stamp) {
          difference_stamp[sz(e)] = stamp;
          difference[sz(e)] = element_size[sz(e)] - weight[sz(i)];
        } else {
          difference[sz(e)] -= weight[sz(i)];
        }
      }
    }

    // Degree bounds, pruning, absorption and mass elimination.
    hashes.clear();
    for (const Int i : new_element) {
      std::vector<Int>& e_list = elems[sz(i)];
      Int bound = 0;
      std::size_t hash = sz(pivot);
      std::size_t kept = 0;
      for (std::size_t p = 0; p < e_list.size(); ++p) {
        const Int e = e_list[p];
        if (state[sz(e)] != Node::kElement) continue;
        const Int outside = difference[sz(e)];
        if (outside > 0) {
          bound += outside;
          hash += sz(e);
          e_list[kept++] = e;
        } else {
          // Every member of e is in the new element: aggressive absorption.
          state[sz(e)] = Node::kAbsorbed;
          release(&members[sz(e)]);
        }
      }
      e_list.resize(kept);
      e_list.insert(e_list.begin(), pivot);

      std::vector<Int>& a_list = vars[sz(i)];
      kept = 0;
      for (std::size_t p = 0; p < a_list.size(); ++p) {
        const Int j = a_list[p];
        // Variables inside the new element are reached through it now, so
        // the direct edge is redundant.
        if (state[sz(j)] != Node::kVariable || flag[sz(j)] == stamp) continue;
        bound += weight[sz(j)];
        hash += sz(j);
        a_list[kept++] = j;
      }
      a_list.resize(kept);

      if (bound == 0) {
        // Nothing outside the new element: i is eliminated with the pivot.
        state[sz(i)] = Node::kGone;
        emit(i);
        eliminated += weight[sz(i)];
        element_weight -= weight[sz(i)];
        weight[sz(i)] = 0;
        release(&vars[sz(i)]);
        release(&elems[sz(i)]);
        continue;
      }
      degree[sz(i)] = std::min(degree[sz(i)], bound);
      hashes.push_back({hash % sz(n), i});
    }

    // Supervariables: variables of the new element with identical element and
    // variable lists. Equal hashes first, then an exact comparison.
    std::sort(hashes.begin(), hashes.end());
    for (std::size_t a = 0; a < hashes.size(); ++a) {
      const Int i = hashes[a].second;
      if (state[sz(i)] != Node::kVariable) continue;
      bool marked = false;
      for (std::size_t b = a + 1; b < hashes.size() && hashes[b].first == hashes[a].first;
           ++b) {
        const Int j = hashes[b].second;
        if (state[sz(j)] != Node::kVariable) continue;
        if (elems[sz(i)].size() != elems[sz(j)].size() ||
            vars[sz(i)].size() != vars[sz(j)].size()) {
          continue;
        }
        if (!marked) {
          ++seen_stamp;
          for (const Int e : elems[sz(i)]) seen[sz(e)] = seen_stamp;
          for (const Int v : vars[sz(i)]) seen[sz(v)] = seen_stamp;
          marked = true;
        }
        bool same = true;
        for (const Int e : elems[sz(j)]) {
          if (seen[sz(e)] != seen_stamp) {
            same = false;
            break;
          }
        }
        for (std::size_t p = 0; same && p < vars[sz(j)].size(); ++p) {
          if (seen[sz(vars[sz(j)][p])] != seen_stamp) same = false;
        }
        if (!same) continue;
        weight[sz(i)] += weight[sz(j)];
        weight[sz(j)] = 0;
        state[sz(j)] = Node::kMerged;
        group[sz(i)].push_back(j);
        for (const Int k : group[sz(j)]) group[sz(i)].push_back(k);
        release(&group[sz(j)]);
        release(&vars[sz(j)]);
        release(&elems[sz(j)]);
      }
    }

    // Final degrees: the bound plus the new element's other members, and
    // never more than what is left to eliminate.
    std::size_t kept = 0;
    for (const Int i : new_element) {
      if (state[sz(i)] != Node::kVariable) continue;
      new_element[kept++] = i;
      Int d = degree[sz(i)] + element_weight - weight[sz(i)];
      d = std::min(d, live - eliminated - weight[sz(i)]);
      d = std::max<Int>(d, 0);
      degree[sz(i)] = d;
      insert(i);
      min_degree = std::min(min_degree, d);
    }
    new_element.resize(kept);
    members[sz(pivot)] = new_element;
    element_size[sz(pivot)] = element_weight;
  }

  for (const Int i : dense) order.push_back(i);
  // Anything the loop could not reach would be a bug above; better a valid
  // permutation with a poor order than an invalid one.
  if (static_cast<Int>(order.size()) != n) {
    std::vector<char> placed(sz(n), 0);
    std::vector<Int> repaired;
    repaired.reserve(sz(n));
    for (const Int i : order) {
      if (i >= 0 && i < n && !placed[sz(i)]) {
        placed[sz(i)] = 1;
        repaired.push_back(i);
      }
    }
    for (Int i = 0; i < n; ++i) {
      if (!placed[sz(i)]) repaired.push_back(i);
    }
    order.swap(repaired);
  }
  return order;
}

bool CholeskyFactor::analyse(const SparseMatrix& lower, const CholeskyOptions& options,
                             std::string* error, const std::vector<Int>* given_perm) {
  factorised_ = false;
  if (lower.rows() != lower.cols()) {
    if (error) *error = "Cholesky needs a square matrix";
    return false;
  }
  n_ = lower.rows();
  input_nonzeros_ = lower.nnz();
  pivot_tolerance_ = options.pivot_tolerance;
  dropped_pivot_ = options.dropped_pivot;
  const Int n = n_;

  // 1. The ordering.
  std::vector<Int> order;
  if (given_perm != nullptr) {
    order = *given_perm;
  } else if (options.ordering == CholeskyOptions::Ordering::kNatural) {
    order.resize(sz(n));
    std::iota(order.begin(), order.end(), 0);
  } else {
    order = approximate_minimum_degree(lower, options);
  }
  if (static_cast<Int>(order.size()) != n) {
    if (error) *error = "the ordering is not a permutation of the rows";
    return false;
  }
  std::vector<Int> inverse(sz(n), -1);
  for (Int k = 0; k < n; ++k) {
    const Int i = order[sz(k)];
    if (i < 0 || i >= n || inverse[sz(i)] >= 0) {
      if (error) *error = "the ordering is not a permutation of the rows";
      return false;
    }
    inverse[sz(i)] = k;
  }
  dense_rows_ = 0;
  if (given_perm == nullptr &&
      options.ordering == CholeskyOptions::Ordering::kApproximateMinimumDegree) {
    const double dense_limit =
        std::fmax(static_cast<double>(options.dense_row_minimum),
                  options.dense_row_factor * std::sqrt(static_cast<double>(n)));
    std::vector<Int> s;
    std::vector<Int> idx;
    symmetric_adjacency(lower, &s, &idx);
    for (Int i = 0; i < n; ++i) {
      if (static_cast<double>(s[sz(i) + 1] - s[sz(i)]) > dense_limit && n > 2) ++dense_rows_;
    }
  }

  // The strictly lower part of each row of P M P', which is what both the
  // elimination tree and the row subtrees are read from.
  auto permuted_rows = [&](const std::vector<Int>& inv, std::vector<Int>* start,
                           std::vector<Int>* index) {
    std::vector<Int> count(sz(n) + 1, 0);
    for (Int i = 0; i < n; ++i) {
      for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
        const Int j = lower.index()[sz(e)];
        if (j >= i) continue;
        const Int a = inv[sz(i)];
        const Int b = inv[sz(j)];
        ++count[sz(std::max(a, b)) + 1];
      }
    }
    for (Int i = 0; i < n; ++i) count[sz(i) + 1] += count[sz(i)];
    index->assign(sz(count[sz(n)]), 0);
    std::vector<Int> cursor(count.begin(), count.end() - 1);
    for (Int i = 0; i < n; ++i) {
      for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
        const Int j = lower.index()[sz(e)];
        if (j >= i) continue;
        const Int a = inv[sz(i)];
        const Int b = inv[sz(j)];
        (*index)[sz(cursor[sz(std::max(a, b))]++)] = std::min(a, b);
      }
    }
    *start = std::move(count);
  };

  // 2. Elimination tree of the permuted matrix, by Liu's algorithm: for each
  // entry (k, i) with i < k, climb from i through the ancestors found so far,
  // compressing the path to k as it goes.
  auto elimination_tree = [&](const std::vector<Int>& start, const std::vector<Int>& index,
                              std::vector<Int>* parent) {
    parent->assign(sz(n), -1);
    std::vector<Int> ancestor(sz(n), -1);
    for (Int k = 0; k < n; ++k) {
      for (Int p = start[sz(k)]; p < start[sz(k) + 1]; ++p) {
        Int i = index[sz(p)];
        while (i != -1 && i < k) {
          const Int up = ancestor[sz(i)];
          ancestor[sz(i)] = k;
          if (up == -1) (*parent)[sz(i)] = k;
          i = up;
        }
      }
    }
  };

  std::vector<Int> row_start;
  std::vector<Int> row_index;
  permuted_rows(inverse, &row_start, &row_index);
  std::vector<Int> parent;
  elimination_tree(row_start, row_index, &parent);

  // 3. Postorder. Descendants then become contiguous and every supernode a run
  // of consecutive columns; the fill does not change.
  {
    std::vector<Int> first_child(sz(n), -1);
    std::vector<Int> sibling(sz(n), -1);
    for (Int j = n; j-- > 0;) {
      if (parent[sz(j)] < 0) continue;
      sibling[sz(j)] = first_child[sz(parent[sz(j)])];
      first_child[sz(parent[sz(j)])] = j;
    }
    std::vector<Int> post;
    post.reserve(sz(n));
    std::vector<Int> stack;
    for (Int root = 0; root < n; ++root) {
      if (parent[sz(root)] >= 0) continue;
      stack.push_back(root);
      while (!stack.empty()) {
        const Int top = stack.back();
        const Int child = first_child[sz(top)];
        if (child >= 0) {
          first_child[sz(top)] = sibling[sz(child)];
          stack.push_back(child);
        } else {
          post.push_back(top);
          stack.pop_back();
        }
      }
    }
    perm_.assign(sz(n), 0);
    for (Int k = 0; k < n; ++k) perm_[sz(k)] = order[sz(post[sz(k)])];
    iperm_.assign(sz(n), 0);
    for (Int k = 0; k < n; ++k) iperm_[sz(perm_[sz(k)])] = k;
  }
  permuted_rows(iperm_, &row_start, &row_index);
  elimination_tree(row_start, row_index, &parent);

  // 4. Column counts from the row subtrees: row k of L is the set of nodes
  // reached climbing from each entry of row k of the matrix until k or an
  // already visited node. One pass, O(nonzeros of L).
  std::vector<Int> column_count(sz(n), 0);
  std::vector<Int> visited(sz(n), -1);
  for (Int k = 0; k < n; ++k) {
    visited[sz(k)] = k;
    for (Int p = row_start[sz(k)]; p < row_start[sz(k) + 1]; ++p) {
      for (Int i = row_index[sz(p)]; visited[sz(i)] != k; i = parent[sz(i)]) {
        visited[sz(i)] = k;
        ++column_count[sz(i)];
      }
    }
  }
  exact_nonzeros_ = n;
  flops_ = 0.0;
  for (Int j = 0; j < n; ++j) {
    exact_nonzeros_ += column_count[sz(j)];
    const double c = static_cast<double>(column_count[sz(j)]) + 1.0;
    flops_ += c * c;
  }

  // 5. Fundamental supernodes, then relaxed amalgamation.
  std::vector<Int> children(sz(n), 0);
  for (Int j = 0; j < n; ++j) {
    if (parent[sz(j)] >= 0) ++children[sz(parent[sz(j)])];
  }
  std::vector<Int> fundamental{0};
  for (Int j = 1; j < n; ++j) {
    const bool chain = parent[sz(j - 1)] == j && children[sz(j)] == 1 &&
                       column_count[sz(j - 1)] == column_count[sz(j)] + 1;
    if (!chain) fundamental.push_back(j);
  }
  fundamental.push_back(n);
  if (n == 0) fundamental.assign(1, 0);

  // Walk from the top of the tree down, merging a supernode into the group
  // that starts right after it when its last column's parent is that group's
  // first column. The merged block's rows are the child's columns followed by
  // the group's rows, because the child's structure below its own columns is
  // inside its parent's, so the explicit zeros it costs are known exactly.
  const Int fundamental_count = static_cast<Int>(fundamental.size()) - 1;
  std::vector<char> joins_next(sz(std::max<Int>(fundamental_count, 0)), 0);
  {
    long long group_cols = 0;
    long long group_rows = 0;
    long long group_zeros = 0;
    bool have_group = false;
    for (Int s = fundamental_count; s-- > 0;) {
      const Int first = fundamental[sz(s)];
      const Int last = fundamental[sz(s) + 1] - 1;
      const long long cols = last - first + 1;
      const long long rows = cols + column_count[sz(last)];
      bool merge = false;
      long long new_cols = 0;
      long long new_rows = 0;
      long long zeros = 0;
      if (have_group && options.relax_columns > 1 && parent[sz(last)] == last + 1) {
        new_cols = cols + group_cols;
        new_rows = cols + group_rows;
        zeros = group_zeros + cols * (new_rows - rows);
        const long long entries = new_cols * new_rows - new_cols * (new_cols - 1) / 2;
        merge = new_cols <= 4 ||
                (new_cols <= options.relax_columns &&
                 static_cast<double>(zeros) <=
                     options.relax_zero_fraction * static_cast<double>(entries));
      }
      if (merge) {
        joins_next[sz(s)] = 1;
        group_cols = new_cols;
        group_rows = new_rows;
        group_zeros = zeros;
      } else {
        group_cols = cols;
        group_rows = rows;
        group_zeros = 0;
        have_group = true;
      }
    }
  }
  super_first_.clear();
  for (Int s = 0; s < fundamental_count; ++s) {
    if (s == 0 || !joins_next[sz(s - 1)]) super_first_.push_back(fundamental[sz(s)]);
  }
  super_first_.push_back(n);
  const Int supers = static_cast<Int>(super_first_.size()) - 1;
  super_of_.assign(sz(n), 0);
  for (Int s = 0; s < supers; ++s) {
    for (Int j = super_first_[sz(s)]; j < super_first_[sz(s) + 1]; ++j) super_of_[sz(j)] = s;
  }

  // 6. Row structure of each supernode: its own columns, then every row k
  // below them that some column of the supernode has an entry in. The row
  // subtrees again, recording instead of counting; rows arrive in order.
  std::vector<std::vector<Int>> below(sz(supers));
  std::vector<Int> last_row(sz(supers), -1);
  std::fill(visited.begin(), visited.end(), -1);
  for (Int k = 0; k < n; ++k) {
    visited[sz(k)] = k;
    for (Int p = row_start[sz(k)]; p < row_start[sz(k) + 1]; ++p) {
      for (Int i = row_index[sz(p)]; visited[sz(i)] != k; i = parent[sz(i)]) {
        visited[sz(i)] = k;
        const Int s = super_of_[sz(i)];
        if (k >= super_first_[sz(s) + 1] && last_row[sz(s)] != k) {
          last_row[sz(s)] = k;
          below[sz(s)].push_back(k);
        }
      }
    }
  }
  row_start_.assign(sz(supers) + 1, 0);
  value_start_.assign(sz(supers) + 1, 0);
  for (Int s = 0; s < supers; ++s) {
    const Int cols = super_first_[sz(s) + 1] - super_first_[sz(s)];
    const Int rows = cols + static_cast<Int>(below[sz(s)].size());
    row_start_[sz(s) + 1] = row_start_[sz(s)] + rows;
    value_start_[sz(s) + 1] =
        value_start_[sz(s)] + static_cast<long long>(rows) * static_cast<long long>(cols);
  }
  rows_.assign(sz(row_start_[sz(supers)]), 0);
  stored_nonzeros_ = 0;
  for (Int s = 0; s < supers; ++s) {
    Int p = row_start_[sz(s)];
    for (Int j = super_first_[sz(s)]; j < super_first_[sz(s) + 1]; ++j) rows_[sz(p++)] = j;
    for (const Int k : below[sz(s)]) rows_[sz(p++)] = k;
    const long long cols = super_first_[sz(s) + 1] - super_first_[sz(s)];
    const long long rows = row_start_[sz(s) + 1] - row_start_[sz(s)];
    stored_nonzeros_ += cols * rows - cols * (cols - 1) / 2;
    std::vector<Int>().swap(below[sz(s)]);
  }

  // 7. Where every entry of the input lands in the dense blocks.
  target_.assign(sz(input_nonzeros_), -1);
  diagonal_of_entry_.assign(sz(input_nonzeros_), -1);
  {
    // Entries grouped by the supernode of their permuted column.
    std::vector<Int> count(sz(supers) + 1, 0);
    for (Int i = 0; i < n; ++i) {
      for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
        const Int j = lower.index()[sz(e)];
        if (j > i) continue;
        const Int column = std::min(iperm_[sz(i)], iperm_[sz(j)]);
        ++count[sz(super_of_[sz(column)]) + 1];
      }
    }
    for (Int s = 0; s < supers; ++s) count[sz(s) + 1] += count[sz(s)];
    std::vector<Int> entries(sz(count[sz(supers)]));
    std::vector<Int> cursor(count.begin(), count.end() - 1);
    for (Int i = 0; i < n; ++i) {
      for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) {
        const Int j = lower.index()[sz(e)];
        if (j > i) continue;
        const Int column = std::min(iperm_[sz(i)], iperm_[sz(j)]);
        entries[sz(cursor[sz(super_of_[sz(column)])]++)] = e;
      }
    }
    // The row of each entry, found from the entry's position: a CSR entry
    // does not know its own row, so record it once.
    std::vector<Int> entry_row(sz(input_nonzeros_), 0);
    for (Int i = 0; i < n; ++i) {
      for (Int e = lower.row_begin(i); e < lower.row_end(i); ++e) entry_row[sz(e)] = i;
    }
    relative_.assign(sz(n), -1);
    for (Int s = 0; s < supers; ++s) {
      const Int first = super_first_[sz(s)];
      const Int height = row_start_[sz(s) + 1] - row_start_[sz(s)];
      for (Int p = row_start_[sz(s)]; p < row_start_[sz(s) + 1]; ++p)
        relative_[sz(rows_[sz(p)])] = p - row_start_[sz(s)];
      for (Int q = count[sz(s)]; q < count[sz(s) + 1]; ++q) {
        const Int e = entries[sz(q)];
        const Int a = iperm_[sz(entry_row[sz(e)])];
        const Int b = iperm_[sz(lower.index()[sz(e)])];
        const Int column = std::min(a, b);
        const Int row = std::max(a, b);
        const Int position = relative_[sz(row)];
        if (position < 0) {
          if (error) *error = "internal error: an entry of the matrix is outside the factor";
          return false;
        }
        target_[sz(e)] = value_start_[sz(s)] +
                         static_cast<long long>(column - first) * height + position;
        if (row == column) diagonal_of_entry_[sz(e)] = column;
      }
      for (Int p = row_start_[sz(s)]; p < row_start_[sz(s) + 1]; ++p)
        relative_[sz(rows_[sz(p)])] = -1;
    }
  }

  values_.assign(static_cast<std::size_t>(value_start_[sz(supers)]), 0.0);
  original_diagonal_.assign(sz(n), 0.0);
  link_head_.assign(sz(supers), -1);
  link_next_.assign(sz(supers), -1);
  next_row_.assign(sz(supers), 0);
  permuted_.assign(sz(n), 0.0);
  return true;
}

bool CholeskyFactor::factorize(const SparseMatrix& lower, std::string* error) {
  if (lower.nnz() != input_nonzeros_ || lower.rows() != n_) {
    if (error) *error = "factorize() was given a matrix with a different pattern";
    return false;
  }
  return factorize(lower.value(), error);
}

bool CholeskyFactor::factorize(const std::vector<double>& input, std::string* error) {
  factorised_ = false;
  if (static_cast<Int>(input.size()) != input_nonzeros_ ||
      target_.size() != input.size()) {
    if (error) *error = "factorize() called before analyse(), or with a different pattern";
    return false;
  }
  const Int supers = supernodes();
  std::fill(values_.begin(), values_.end(), 0.0);
  std::fill(original_diagonal_.begin(), original_diagonal_.end(), 0.0);
  for (std::size_t e = 0; e < input.size(); ++e) {
    if (target_[e] < 0) continue;
    values_[static_cast<std::size_t>(target_[e])] += input[e];
    if (diagonal_of_entry_[e] >= 0) original_diagonal_[sz(diagonal_of_entry_[e])] += input[e];
  }
  std::fill(link_head_.begin(), link_head_.end(), -1);
  dropped_pivots_ = 0;

  for (Int s = 0; s < supers; ++s) {
    const Int first = super_first_[sz(s)];
    const Int last = super_first_[sz(s) + 1] - 1;
    const Int cols = last - first + 1;
    const Int height = row_start_[sz(s) + 1] - row_start_[sz(s)];
    const Int* my_rows = rows_.data() + row_start_[sz(s)];
    double* block = values_.data() + value_start_[sz(s)];
    for (Int p = 0; p < height; ++p) relative_[sz(my_rows[p])] = p;

    // Updates from every supernode below whose structure reaches into these
    // columns. Each one is a dense product of two slices of its block,
    // subtracted through the relative indices.
    Int k = link_head_[sz(s)];
    link_head_[sz(s)] = -1;
    while (k >= 0) {
      const Int following = link_next_[sz(k)];
      const Int k_cols = super_first_[sz(k) + 1] - super_first_[sz(k)];
      const Int k_height = row_start_[sz(k) + 1] - row_start_[sz(k)];
      const Int* k_rows = rows_.data() + row_start_[sz(k)];
      const double* k_block = values_.data() + value_start_[sz(k)];
      const Int p1 = next_row_[sz(k)];
      Int p2 = p1;
      while (p2 < k_height && k_rows[p2] <= last) ++p2;
      const Int span = p2 - p1;          // columns of s touched
      const Int tall = k_height - p1;    // rows of s touched
      const std::size_t need = sz(tall) * sz(span);
      if (update_.size() < need) update_.resize(need);
      std::fill(update_.begin(), update_.begin() + static_cast<std::ptrdiff_t>(need), 0.0);
      for (Int c = 0; c < k_cols; ++c) {
        const double* column = k_block + static_cast<long long>(c) * k_height;
        for (Int b = p1; b < p2; ++b) {
          const double lb = column[b];
          if (lb == 0.0) continue;
          double* out = update_.data() + sz(b - p1) * sz(tall);
          for (Int a = b; a < k_height; ++a) out[a - p1] += column[a] * lb;
        }
      }
      for (Int b = p1; b < p2; ++b) {
        double* dest = block + static_cast<long long>(k_rows[b] - first) * height;
        const double* from = update_.data() + sz(b - p1) * sz(tall);
        for (Int a = b; a < k_height; ++a) dest[relative_[sz(k_rows[a])]] -= from[a - p1];
      }
      next_row_[sz(k)] = p2;
      if (p2 < k_height) {
        const Int target = super_of_[sz(k_rows[p2])];
        link_next_[sz(k)] = link_head_[sz(target)];
        link_head_[sz(target)] = k;
      }
      k = following;
    }

    // Dense Cholesky of the diagonal block, carrying the rows below along.
    for (Int c = 0; c < cols; ++c) {
      double* column = block + static_cast<long long>(c) * height;
      double pivot = column[c];
      const double reference = std::fabs(original_diagonal_[sz(first + c)]);
      if (!(pivot > pivot_tolerance_ * reference) || !(pivot > 0.0) || !std::isfinite(pivot)) {
        pivot = dropped_pivot_;
        ++dropped_pivots_;
      }
      const double root = std::sqrt(pivot);
      column[c] = root;
      const double inverse = 1.0 / root;
      for (Int a = c + 1; a < height; ++a) column[a] *= inverse;
      for (Int c2 = c + 1; c2 < cols; ++c2) {
        const double l = column[c2];
        if (l == 0.0) continue;
        double* target = block + static_cast<long long>(c2) * height;
        for (Int a = c2; a < height; ++a) target[a] -= column[a] * l;
      }
    }
    for (Int p = 0; p < height; ++p) relative_[sz(my_rows[p])] = -1;

    if (height > cols) {
      next_row_[sz(s)] = cols;
      const Int target = super_of_[sz(my_rows[cols])];
      link_next_[sz(s)] = link_head_[sz(target)];
      link_head_[sz(target)] = s;
    }
  }
  factorised_ = true;
  return true;
}

void CholeskyFactor::solve(std::vector<double>* x) const {
  if (static_cast<Int>(x->size()) != n_) return;
  solve(x->data());
}

void CholeskyFactor::solve(double* x) const {
  if (!factorised_) return;
  const Int n = n_;
  double* y = permuted_.data();
  for (Int k = 0; k < n; ++k) y[k] = x[perm_[sz(k)]];
  const Int supers = supernodes();
  // L z = y, a supernode at a time: the triangle, then the rectangle below.
  for (Int s = 0; s < supers; ++s) {
    const Int first = super_first_[sz(s)];
    const Int cols = super_first_[sz(s) + 1] - first;
    const Int height = row_start_[sz(s) + 1] - row_start_[sz(s)];
    const Int* my_rows = rows_.data() + row_start_[sz(s)];
    const double* block = values_.data() + value_start_[sz(s)];
    for (Int c = 0; c < cols; ++c) {
      const double* column = block + static_cast<long long>(c) * height;
      const double v = y[first + c] / column[c];
      y[first + c] = v;
      if (v == 0.0) continue;
      for (Int a = c + 1; a < height; ++a) y[my_rows[a]] -= column[a] * v;
    }
  }
  // L' x = z, in reverse.
  for (Int s = supers; s-- > 0;) {
    const Int first = super_first_[sz(s)];
    const Int cols = super_first_[sz(s) + 1] - first;
    const Int height = row_start_[sz(s) + 1] - row_start_[sz(s)];
    const Int* my_rows = rows_.data() + row_start_[sz(s)];
    const double* block = values_.data() + value_start_[sz(s)];
    for (Int c = cols; c-- > 0;) {
      const double* column = block + static_cast<long long>(c) * height;
      double sum = y[first + c];
      for (Int a = c + 1; a < height; ++a) sum -= column[a] * y[my_rows[a]];
      y[first + c] = sum / column[c];
    }
  }
  for (Int k = 0; k < n; ++k) x[perm_[sz(k)]] = y[k];
}

}  // namespace anukulan
