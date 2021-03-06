from typing import Tuple


def parse_aliases(statements: str, join_count: int, ordered: bool) -> Tuple[list, list, int]:
    aliases = []
    predicates = []
    statements = statements.split(",")

    for r in statements:
        r = r.replace("\n", '').strip()
        if " join " in r:
            joins = r.split(" join ")
            for join in joins:
                join = join.strip()
                if " on " in join:
                    preds = join.split(" on ")
                    for pred in preds:
                        if " as " in pred:
                            rel_name, alias = pred.split(" as ")
                            aliases.append((rel_name.strip(), alias.strip()))
                        else:
                            pred_spl = pred.split()
                            lhs = pred_spl[0]

                            if lhs[0] == "(":
                                lhs = lhs[1:]

                            if "." not in lhs:
                                # BETWEEN
                                continue
                            try:
                                if len(pred_spl) == 2:
                                    combined = pred_spl[1]
                                    combined = combined.replace("'", " ")
                                    combined = combined.split()
                                    val = [combined[1]]
                                else:
                                    val = pred_spl[2:]
                                if "super" in val:
                                    import pdb
                                    pdb.set_trace()
                            except:
                                print("Failed parse join statement: {0}".format(r))
                                continue

                            rel_alias, attr = lhs.split(".")

                            if "=" in pred:
                                rhs = pred.split()[-1]
                                if "." in rhs:
                                    right_rel_alias, right_attr = rhs.split(".")
                                    predicates.append(
                                        (f"join{('_' + str(join_count)) if ordered else ''}", (rel_alias, attr, right_rel_alias, right_attr))
                                    )
                                    join_count += 1
                                    continue
                elif " as " in join:
                    rel_name, alias = join.split(" as ")
                    aliases.append((rel_name.strip(), alias.strip()))
                else:
                    aliases.append((join.strip(), join.strip()))

        else:
            rel_name = r
            alias = r
            if " as " in r:
                rel_name, alias = r.split(" as ")
            aliases.append((rel_name.strip(), alias.strip()))

    return aliases, predicates, join_count


def parse_trailing_operations(clause: str):
    group_by = ''
    having = ''
    sorting = ''
    parsed_clause = clause
    if " group by " in clause:
        parsed_clause, group_by = clause.split(" group by ")

    if " having " in clause:
        if " having " in group_by:
            group_by, having = group_by.split(" having ")
        else:
            parsed_clause, having = clause.split(" having ")

    if " order by " in clause:
        if " order by " in having:
            having, sorting = having.split(" order by ")
        elif " order by " in group_by:
            group_by, sorting = group_by.split(" order by ")
        else:
            parsed_clause, sorting = clause.split(" order by ")

    return parsed_clause, group_by, having, sorting


class JOBQuery:
    def __init__(self, query, ordered=False):
        query = query.replace("\n", " ").lower().rstrip()
        self.original_sql = query
        self.ordered = ordered

        if query.endswith(";"):
            query = query[:-1]

        projs = query.split(" from ")[0][7:]

        from_clause = query.split(" from ")[1].split(" where ")[0] if " where " in query else query.split(" from ")[1]
        from_clause, group_by_clause, having_clause, sorting_clause = parse_trailing_operations(from_clause)

        where = query.split(" where ")[-1] if " where " in query else ''
        where, group_by_clause, having_clause, sorting_clause = parse_trailing_operations(where)

        self.join_count = 1
        self.scan_count = 1
        self.predicates = []
        self.__original_where = where
        self.__parse_from(from_clause)
        self.__parse_projs(projs)
        if len(where) > 0:
            self.__parse_where(where)

        self.rel_lookup = {y: x for (x, y) in self.__relations}

        self.__join_edges = {}
        for (_, lh), (_, rh) in self.joins(with_attrs=False):
            if lh not in self.__join_edges:
                self.__join_edges[lh] = set()

            if rh not in self.__join_edges:
                self.__join_edges[rh] = set()

            self.__join_edges[lh].add(rh)
            self.__join_edges[rh].add(lh)

    def relations(self):
        return sorted(self.rel_lookup.keys())

    def tables(self):
        return sorted(self.rel_lookup.values())

    def table_for_relation(self, rel):
        return self.rel_lookup[rel]

    def joins_with(self, rel_alias):
        return self.__join_edges[rel_alias]

    def joins(self, with_attrs=True):
        for (pt, pv) in self.predicates:
            if pt != " join ":
                continue

            lh, lha, rh, rha = pv
            if with_attrs:
                yield self.rel_lookup[lh], lha, self.rel_lookup[rh], rha
            else:
                yield (self.rel_lookup[lh], lh), (self.rel_lookup[rh], rh)

    def attrs_with_predicate(self, values=False):
        for (pt, pv) in self.predicates:
            if pt != "scan":
                continue

            lh, lha, cmp_op, val = pv
            if values:
                yield self.rel_lookup[lh], lha, cmp_op, val
            else:
                yield self.rel_lookup[lh], lha

    def reconstruct_where(self):
        return self.__original_where

    def __parse_from(self, from_clause):
        self.__relations, self.predicates, self.join_count = parse_aliases(from_clause, self.join_count, self.ordered)

    def __parse_projs(self, projs):
        self.projs, predicates, self.join_count = parse_aliases(projs, self.join_count, self.ordered)
        self.predicates += predicates

    def __parse_where(self, where):
        preds = where.split(" and ")

        for pred in preds:
            pred_spl = pred.split()
            lhs = pred_spl[0]

            if lhs[0] == "(":
                lhs = lhs[1:]

            if "." not in lhs:
                # BETWEEN
                continue
            try:
                if len(pred_spl) == 2:
                    combined = pred_spl[1]
                    combined = combined.replace("'", " ")
                    combined = combined.split()
                    cmp_op = combined[0]
                    # make it a list
                    val = [combined[1]]
                else:
                    cmp_op = pred_spl[1]
                    val = pred_spl[2:]
                if "super" in val:
                    import pdb
                    pdb.set_trace()
            except:
                print("Failed parse where statement: {0}".format(where))
                continue

            rel_alias, attr = lhs.split(".")

            if "=" in pred:
                rhs = pred.split()[-1]
                if "'" in rhs:
                    val = rhs
                elif "." in rhs:
                    right_rel_alias, right_attr = rhs.split(".")
                    self.predicates.append(
                        (f"join{('_' + str(self.join_count)) if self.ordered else ''}", (rel_alias, attr, right_rel_alias, right_attr))
                    )
                    self.join_count += 1
                    continue
            self.predicates.append((f"scan{('_' + str(self.scan_count)) if self.ordered else ''}", (rel_alias, attr, cmp_op, val[0])))
            self.scan_count += 1
