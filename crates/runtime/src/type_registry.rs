use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ptr::NonNull,
};

use crate::managed_rc::{ChildrenVisitorFn, RcHeader};

type TypeEntryFns = (
    ChildrenVisitorFn,
    fn(NonNull<RcHeader>),
    fn(NonNull<RcHeader>),
);

struct TypeEntry {
    children: ChildrenVisitorFn,
    clear_cycle_fields: fn(NonNull<RcHeader>),
    dropper: fn(NonNull<RcHeader>),
}

thread_local! {
    static TYPE_REGISTRY: RefCell<HashMap<u32, TypeEntry>> = RefCell::new(HashMap::new());
    static CYCLE_CAPABLE: RefCell<HashSet<u32>> = RefCell::new(HashSet::new());
}

/// Called once per dataref type at program startup
pub fn register_cycle_capable(type_id: u32) {
    CYCLE_CAPABLE.with(|set| {
        set.borrow_mut().insert(type_id);
    });
}

pub fn is_cycle_capable(type_id: u32) -> bool {
    CYCLE_CAPABLE.with(|set| set.borrow().contains(&type_id))
}

/// Does not mark the type as cycle-capable, call `register_cycle_capable` separately
pub fn register_child_traverser(
    type_id: u32,
    children: ChildrenVisitorFn,
    clear_cycle_fields: fn(NonNull<RcHeader>),
    dropper: fn(NonNull<RcHeader>),
) {
    TYPE_REGISTRY.with(|reg| {
        reg.borrow_mut().insert(
            type_id,
            TypeEntry {
                children,
                clear_cycle_fields,
                dropper,
            },
        );
    });
}

pub fn get_type_entry(type_id: u32) -> Option<TypeEntryFns> {
    TYPE_REGISTRY.with(|reg| {
        reg.borrow()
            .get(&type_id)
            .map(|e| (e.children, e.clear_cycle_fields, e.dropper))
    })
}

#[cfg(test)]
pub fn clear_registry() {
    TYPE_REGISTRY.with(|reg| reg.borrow_mut().clear());
    CYCLE_CAPABLE.with(|set| set.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::{
        cycle_collector::{clear_suspects, collect_cycles, reset_collect_threshold, suspect_count},
        managed_rc::{CycleVtable, ManagedRc, ManagedRcInner, typed_dropper},
    };

    fn no_children(_: NonNull<RcHeader>, _: &mut dyn FnMut(NonNull<RcHeader>)) {}
    fn no_clear(_: NonNull<RcHeader>) {}
    fn no_dropper(_: NonNull<RcHeader>) {}

    #[test]
    fn register_and_query_cycle_capable() {
        clear_registry();
        register_cycle_capable(1);
        assert!(is_cycle_capable(1));
        assert!(!is_cycle_capable(2));
        clear_registry();
    }

    #[test]
    fn register_child_traverser_and_retrieve() {
        clear_registry();
        register_child_traverser(1, no_children, no_clear, no_dropper);
        assert!(get_type_entry(1).is_some());
        assert!(get_type_entry(2).is_none());
        let (children, clear, dropper) = get_type_entry(1).unwrap();
        assert_eq!(
            children as *const () as usize,
            no_children as *const () as usize
        );
        assert_eq!(clear as *const () as usize, no_clear as *const () as usize);
        assert_eq!(
            dropper as *const () as usize,
            no_dropper as *const () as usize
        );
        clear_registry();
    }

    #[test]
    fn double_registration_is_idempotent() {
        clear_registry();
        register_cycle_capable(1);
        register_cycle_capable(1);
        assert!(is_cycle_capable(1));
        clear_registry();
    }

    #[test]
    fn register_traverser_does_not_imply_cycle_capable() {
        clear_registry();
        register_child_traverser(1, no_children, no_clear, no_dropper);
        assert!(!is_cycle_capable(1));
        register_cycle_capable(1);
        assert!(is_cycle_capable(1));
        clear_registry();
    }

    struct Node {
        next: RefCell<Option<ManagedRc<Node>>>,
    }

    fn node_children(ptr: NonNull<RcHeader>, f: &mut dyn FnMut(NonNull<RcHeader>)) {
        let data = unsafe { &(*ptr.cast::<ManagedRcInner<Node>>().as_ptr()).data };
        if let Some(child) = data.next.borrow().as_ref() {
            f(child.header_ptr());
        }
    }

    fn node_clear(ptr: NonNull<RcHeader>) {
        let data = unsafe { &mut (*ptr.cast::<ManagedRcInner<Node>>().as_ptr()).data };
        let maybe_child = data.next.borrow_mut().take();
        if let Some(child) = maybe_child {
            std::mem::forget(child);
        }
    }

    const NODE_TYPE_ID: u32 = 42;

    static NODE_VT: CycleVtable = CycleVtable {
        type_name: "Node",
        children: node_children,
        clear_cycle_fields: node_clear,
        dropper: typed_dropper::<Node>,
        buffer_on_decrement: true,
    };

    fn setup_node_type() {
        register_cycle_capable(NODE_TYPE_ID);
        register_child_traverser(
            NODE_TYPE_ID,
            node_children,
            node_clear,
            typed_dropper::<Node>,
        );
    }

    #[test]
    fn managed_rc_cycle_through_public_api() {
        clear_registry();
        clear_suspects();
        reset_collect_threshold();
        setup_node_type();

        assert!(is_cycle_capable(NODE_TYPE_ID));
        assert!(get_type_entry(NODE_TYPE_ID).is_some());

        let a = ManagedRc::new_with_vtable(
            Node {
                next: RefCell::new(None),
            },
            &NODE_VT,
        );
        let b = ManagedRc::new_with_vtable(
            Node {
                next: RefCell::new(None),
            },
            &NODE_VT,
        );

        *a.next.borrow_mut() = Some(b.clone());
        *b.next.borrow_mut() = Some(a.clone());

        assert_eq!(a.strong_count(), 2);
        assert_eq!(b.strong_count(), 2);

        // normal drop triggers cycle-aware path via vtable
        drop(a); // strong 2 -> 1, buffered
        drop(b); // strong 2 -> 1, buffered

        assert_eq!(suspect_count(), 2);
        collect_cycles();
        assert_eq!(suspect_count(), 0);

        clear_registry();
    }
}
