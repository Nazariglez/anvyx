pub mod cycle_collector;
pub mod managed_rc;
pub mod suspect_buffer;
pub mod type_registry;

pub use managed_rc::{CycleColor, CycleVtable, ManagedRc, ManagedRcInner, RcHeader, typed_dropper, managed_alloc_count, managed_alloc_details};
pub use cycle_collector::{collect_cycles, set_auto_collect};
pub use type_registry::{
    get_type_entry, is_cycle_capable, register_child_traverser, register_cycle_capable,
};
