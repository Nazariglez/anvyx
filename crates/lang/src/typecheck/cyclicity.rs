use std::collections::{HashMap, HashSet, VecDeque};

use super::{
    infer::{build_subst, subst_type},
    types::{EnumDef, StructDef},
    visit::type_any,
};
use crate::ast::{Ident, Type, VariantKind};

pub(super) fn analyze_cyclicity(
    struct_defs: &HashMap<Ident, StructDef>,
    enum_defs: &HashMap<Ident, EnumDef>,
) -> HashSet<Ident> {
    let datarefs: Vec<Ident> = struct_defs
        .iter()
        .filter(|(_, def)| def.is_dataref)
        .map(|(name, _)| *name)
        .collect();

    if datarefs.is_empty() {
        return HashSet::new();
    }

    let mut adj: HashMap<Ident, Vec<Ident>> = HashMap::new();
    for &name in &datarefs {
        let def = &struct_defs[&name];
        let mut edges = vec![];
        for field in &def.fields {
            extract_dataref_refs(&field.ty, struct_defs, enum_defs, &mut edges);
        }
        adj.insert(name, edges);
    }

    let mut state = TarjanState {
        index_counter: 0,
        stack: vec![],
        on_stack: HashSet::new(),
        indices: HashMap::new(),
        lowlinks: HashMap::new(),
        sccs: vec![],
        adj,
    };

    for &name in &datarefs {
        if !state.indices.contains_key(&name) {
            state.strongconnect(name);
        }
    }

    // cycle-capable, multi-node SCC or a single node with a self-loop
    let mut cycle_core: HashSet<Ident> = HashSet::new();
    for scc in &state.sccs {
        let is_cycle_capable = scc.len() > 1
            || scc
                .first()
                .is_some_and(|v| state.adj.get(v).is_some_and(|nb| nb.contains(v)));
        if is_cycle_capable {
            cycle_core.extend(scc.iter().copied());
        }
    }

    // bfs backwards on the reverse graph, any type that can reach a cycle-capable node is itself cycle-capable
    let mut reverse_adj: HashMap<Ident, Vec<Ident>> = HashMap::new();
    for (src, dests) in &state.adj {
        for &dest in dests {
            reverse_adj.entry(dest).or_default().push(*src);
        }
    }

    let mut cycle_capable = cycle_core.clone();
    let mut queue: VecDeque<Ident> = cycle_core.iter().copied().collect();

    // any dataref with a fn(...) field is unconditionally cycle-capable, a closure stored
    // in it might capture a reference back to the owning dataref at runtime
    for &name in &datarefs {
        let def = &struct_defs[&name];
        let has_fn_field = def
            .fields
            .iter()
            .any(|f| type_contains_func(&f.ty, enum_defs));
        if has_fn_field && cycle_capable.insert(name) {
            queue.push_back(name);
        }
    }

    while let Some(node) = queue.pop_front() {
        if let Some(predecessors) = reverse_adj.get(&node) {
            for &pred in predecessors {
                if cycle_capable.insert(pred) {
                    queue.push_back(pred);
                }
            }
        }
    }

    cycle_capable
}

fn extract_dataref_refs(
    ty: &Type,
    struct_defs: &HashMap<Ident, StructDef>,
    enum_defs: &HashMap<Ident, EnumDef>,
    out: &mut Vec<Ident>,
) {
    type_any(ty, &mut |t| {
        match t {
            Type::DataRef { name, .. } | Type::Struct { name, .. } | Type::UnresolvedName(name) => {
                if struct_defs.get(name).is_some_and(|d| d.is_dataref) {
                    out.push(*name);
                }
            }
            Type::Enum { name, type_args } => {
                // type_any handles type_args recursion automatically
                // expand variant fields which are not part of the structural tree
                if let Some(enum_def) = enum_defs.get(name) {
                    let subst = build_subst(&enum_def.type_params, type_args);
                    for variant in &enum_def.variants {
                        match &variant.kind {
                            VariantKind::Unit => {}
                            VariantKind::Tuple(tys) => {
                                for field_ty in tys {
                                    let resolved = subst_type(field_ty, &subst);
                                    extract_dataref_refs(&resolved, struct_defs, enum_defs, out);
                                }
                            }
                            VariantKind::Struct(fields) => {
                                for field in fields {
                                    let resolved = subst_type(&field.ty, &subst);
                                    extract_dataref_refs(&resolved, struct_defs, enum_defs, out);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        false // never short-circuit, collecting, not searching
    });
}

fn type_contains_func(ty: &Type, enum_defs: &HashMap<Ident, EnumDef>) -> bool {
    type_any(ty, &mut |t| match t {
        Type::Func { .. } => true,
        Type::Enum { name, type_args } => {
            // type_any handles type_args recursion automatically
            // expand variant fields which are not part of the structural tree
            enum_defs.get(name).is_some_and(|def| {
                let subst = build_subst(&def.type_params, type_args);
                def.variants.iter().any(|v| match &v.kind {
                    VariantKind::Unit => false,
                    VariantKind::Tuple(tys) => tys.iter().any(|t| {
                        let resolved = subst_type(t, &subst);
                        type_contains_func(&resolved, enum_defs)
                    }),
                    VariantKind::Struct(fields) => fields.iter().any(|f| {
                        let resolved = subst_type(&f.ty, &subst);
                        type_contains_func(&resolved, enum_defs)
                    }),
                })
            })
        }
        _ => false,
    })
}

struct TarjanState {
    index_counter: u32,
    stack: Vec<Ident>,
    on_stack: HashSet<Ident>,
    indices: HashMap<Ident, u32>,
    lowlinks: HashMap<Ident, u32>,
    sccs: Vec<Vec<Ident>>,
    adj: HashMap<Ident, Vec<Ident>>,
}

impl TarjanState {
    fn strongconnect(&mut self, v: Ident) {
        self.indices.insert(v, self.index_counter);
        self.lowlinks.insert(v, self.index_counter);
        self.index_counter += 1;
        self.stack.push(v);
        self.on_stack.insert(v);

        let neighbors = self.adj.get(&v).cloned().unwrap_or_default();
        for w in neighbors {
            if !self.indices.contains_key(&w) {
                self.strongconnect(w);
                let lowlink_w = self.lowlinks[&w];
                let lowlink_v = self.lowlinks[&v];
                self.lowlinks.insert(v, lowlink_v.min(lowlink_w));
            } else if self.on_stack.contains(&w) {
                let index_w = self.indices[&w];
                let lowlink_v = self.lowlinks[&v];
                self.lowlinks.insert(v, lowlink_v.min(index_w));
            }
        }

        if self.lowlinks[&v] == self.indices[&v] {
            let mut scc = vec![];
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack.remove(&w);
                scc.push(w);
                if w == v {
                    break;
                }
            }
            self.sccs.push(scc);
        }
    }
}
