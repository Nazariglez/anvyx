use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

use crate::cycle_collector::CollectStats;
use crate::managed_rc::RcHeader;

pub(crate) const MIN_THRESHOLD: usize = 256;
pub(crate) const MAX_THRESHOLD: usize = 4096;
const DEFAULT_THRESHOLD: usize = 256;

pub struct SuspectEntry {
    pub ptr: NonNull<RcHeader>,
}

thread_local! {
    pub(crate) static SUSPECT_BUFFER: RefCell<Vec<SuspectEntry>> = RefCell::new(vec![]);
    static AUTO_COLLECT_ENABLED: Cell<bool> = Cell::new(true);
    static COLLECT_THRESHOLD: Cell<usize> = Cell::new(DEFAULT_THRESHOLD);
}

pub fn set_auto_collect(enabled: bool) {
    AUTO_COLLECT_ENABLED.with(|flag| flag.set(enabled));
}

pub fn push_suspect(ptr: NonNull<RcHeader>) {
    SUSPECT_BUFFER.with(|buf| {
        buf.borrow_mut().push(SuspectEntry { ptr });
    });
    maybe_collect();
}

pub fn maybe_collect() {
    let threshold = COLLECT_THRESHOLD.with(|t| t.get());
    let should = AUTO_COLLECT_ENABLED.with(|flag| flag.get())
        && SUSPECT_BUFFER.with(|buf| buf.borrow().len() >= threshold);
    if should {
        let stats = crate::cycle_collector::collect_cycles();
        adjust_threshold(stats);
    }
}

fn adjust_threshold(stats: CollectStats) {
    if stats.suspects == 0 {
        return;
    }
    let effectiveness = stats.freed as f64 / stats.suspects as f64;
    COLLECT_THRESHOLD.with(|t| {
        let current = t.get();
        if effectiveness >= 0.5 {
            t.set((current * 2 / 3).max(MIN_THRESHOLD));
        } else if effectiveness < 0.1 {
            t.set((current * 3 / 2).min(MAX_THRESHOLD));
        }
    });
}

pub fn suspect_count() -> usize {
    SUSPECT_BUFFER.with(|buf| buf.borrow().len())
}

pub fn clear_suspects() {
    SUSPECT_BUFFER.with(|buf| buf.borrow_mut().clear());
}

pub fn get_collect_threshold() -> usize {
    COLLECT_THRESHOLD.with(|t| t.get())
}

pub fn reset_collect_threshold() {
    COLLECT_THRESHOLD.with(|t| t.set(DEFAULT_THRESHOLD));
}

pub fn set_collect_threshold(val: usize) {
    COLLECT_THRESHOLD.with(|t| t.set(val));
}
