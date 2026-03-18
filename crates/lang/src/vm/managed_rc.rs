use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleColor {
    Black,
    Purple,
    Gray,
    White,
}

pub struct RcHeader {
    strong: Cell<u32>,
    color: Cell<CycleColor>,
    buffered: Cell<bool>,
}

impl RcHeader {
    fn new() -> Self {
        Self {
            strong: Cell::new(1),
            color: Cell::new(CycleColor::Black),
            buffered: Cell::new(false),
        }
    }
}

struct ManagedRcInner<T> {
    header: RcHeader,
    data: T,
}

pub struct ManagedRc<T> {
    ptr: NonNull<ManagedRcInner<T>>,
}

impl<T> ManagedRc<T> {
    pub fn new(data: T) -> Self {
        let inner = Box::new(ManagedRcInner {
            header: RcHeader::new(),
            data,
        });
        Self {
            // SAFETY: Box::into_raw never returns null.
            ptr: unsafe { NonNull::new_unchecked(Box::into_raw(inner)) },
        }
    }

    pub fn strong_count(&self) -> u32 {
        self.inner().header.strong.get()
    }

    /// Returns a mutable reference to the inner data.
    ///
    /// If this is the sole owner (`strong == 1`), the reference points directly
    /// into the existing allocation. If the data is shared (`strong > 1`), the
    /// data is cloned into a fresh allocation first (COW), then the mutable
    /// reference into that new allocation is returned. The caller's handle is
    /// silently reseated to the new allocation.
    pub fn make_mut(&mut self) -> &mut T
    where
        T: Clone,
    {
        if self.strong_count() > 1 {
            *self = ManagedRc::new((**self).clone());
        }
        // SAFETY: strong == 1, &mut self guarantees exclusive access.
        unsafe { &mut self.ptr.as_mut().data }
    }

    /// Returns `true` when both handles point to the same heap allocation.
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.ptr == b.ptr
    }

    fn inner(&self) -> &ManagedRcInner<T> {
        // SAFETY: ptr is valid while any ManagedRc handle exists.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> Clone for ManagedRc<T> {
    fn clone(&self) -> Self {
        let header = &self.inner().header;
        let count = header.strong.get();
        if count == u32::MAX {
            std::process::abort();
        }
        header.strong.set(count + 1);
        Self { ptr: self.ptr }
    }
}

impl<T> Drop for ManagedRc<T> {
    fn drop(&mut self) {
        let count = {
            let header = &self.inner().header;
            let c = header.strong.get();
            debug_assert!(c > 0, "ManagedRc: double-free (strong was already 0)");
            header.strong.set(c - 1);
            c
        };
        if count == 1 {
            // SAFETY: last handle; Box reconstructs and frees exactly once.
            unsafe { drop(Box::from_raw(self.ptr.as_ptr())) }
        }
    }
}

impl<T> Deref for ManagedRc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: ptr is always valid for the lifetime of the ManagedRc.
        unsafe { &self.ptr.as_ref().data }
    }
}

impl<T: fmt::Debug> fmt::Debug for ManagedRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display> fmt::Display for ManagedRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for ManagedRc<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for ManagedRc<T> {}

impl<T: PartialOrd> PartialOrd for ManagedRc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: Ord> Ord for ManagedRc<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: Hash> Hash for ManagedRc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct DropTracker<'a> {
        dropped: &'a Cell<bool>,
    }

    impl Drop for DropTracker<'_> {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }

    #[test]
    fn new_sets_strong_count_to_one() {
        let rc = ManagedRc::new(42);
        assert_eq!(rc.strong_count(), 1);
    }

    #[test]
    fn new_data_accessible_via_deref() {
        let rc = ManagedRc::new(42);
        assert_eq!(*rc, 42);
    }

    #[test]
    fn clone_increments_strong_count() {
        let a = ManagedRc::new(42);
        let b = a.clone();
        assert_eq!(a.strong_count(), 2);
        assert_eq!(b.strong_count(), 2);
    }

    #[test]
    fn clone_shares_same_data() {
        let a = ManagedRc::new(42);
        let b = a.clone();
        assert_eq!(*a, *b);
        assert!(ManagedRc::ptr_eq(&a, &b));
    }

    #[test]
    fn multiple_clones_track_count() {
        let a = ManagedRc::new(42);
        let b = a.clone();
        let c = a.clone();
        let d = a.clone();
        assert_eq!(a.strong_count(), 4);
        assert_eq!(b.strong_count(), 4);
        assert_eq!(c.strong_count(), 4);
        assert_eq!(d.strong_count(), 4);
    }

    #[test]
    fn drop_frees_at_zero() {
        let dropped = Cell::new(false);
        {
            let _rc = ManagedRc::new(DropTracker { dropped: &dropped });
        }
        assert!(dropped.get());
    }

    #[test]
    fn drop_does_not_free_while_shared() {
        let dropped = Cell::new(false);
        let rc = ManagedRc::new(DropTracker { dropped: &dropped });
        let clone = rc.clone();
        drop(clone);
        assert!(!dropped.get());
        assert_eq!(rc.strong_count(), 1);
        drop(rc);
        assert!(dropped.get());
    }

    #[test]
    fn drop_decrements_count() {
        let rc = ManagedRc::new(42i32);
        let clone1 = rc.clone();
        let clone2 = rc.clone();
        assert_eq!(rc.strong_count(), 3);
        drop(clone1);
        assert_eq!(rc.strong_count(), 2);
        drop(clone2);
        assert_eq!(rc.strong_count(), 1);
    }

    #[test]
    fn make_mut_unique_returns_mutable_ref() {
        let mut rc = ManagedRc::new(vec![1, 2]);
        rc.make_mut().push(3);
        assert_eq!(*rc, vec![1, 2, 3]);
    }

    #[test]
    fn make_mut_unique_does_not_reallocate() {
        let mut rc = ManagedRc::new(42i32);
        let ptr_before = rc.ptr.as_ptr();
        let _ = rc.make_mut();
        assert_eq!(rc.ptr.as_ptr(), ptr_before);
    }

    #[test]
    fn make_mut_shared_clones_data() {
        let mut a = ManagedRc::new(42i32);
        let b = a.clone();
        assert_eq!(a.strong_count(), 2);
        let _ = a.make_mut();
        assert!(!ManagedRc::ptr_eq(&a, &b));
        assert_eq!(a.strong_count(), 1);
        assert_eq!(b.strong_count(), 1);
    }

    #[test]
    fn make_mut_shared_preserves_original() {
        let mut a = ManagedRc::new(vec![1, 2]);
        let b = a.clone();
        a.make_mut().push(3);
        assert_eq!(*a, vec![1, 2, 3]);
        assert_eq!(*b, vec![1, 2]);
    }

    #[test]
    fn make_mut_multiple_sharers() {
        let mut a = ManagedRc::new(42i32);
        let b = a.clone();
        let c = a.clone();
        let d = a.clone();
        assert_eq!(a.strong_count(), 4);
        let _ = a.make_mut();
        assert_eq!(a.strong_count(), 1);
        assert_eq!(b.strong_count(), 3);
        assert_eq!(c.strong_count(), 3);
        assert_eq!(d.strong_count(), 3);
    }

    #[test]
    fn make_mut_string_cow() {
        let mut a = ManagedRc::new(String::from("hello"));
        let b = a.clone();
        a.make_mut().push_str(" world");
        assert_eq!(&*a, "hello world");
        assert_eq!(&*b, "hello");
    }

    #[test]
    fn debug_forwards_to_inner() {
        assert_eq!(format!("{:?}", ManagedRc::new(42)), format!("{:?}", 42));
    }

    #[test]
    fn display_forwards_to_inner() {
        let rc = ManagedRc::new(String::from("hello"));
        assert!(format!("{rc}").contains("hello"));
    }

    #[test]
    fn partial_eq_compares_values() {
        assert_eq!(ManagedRc::new(42), ManagedRc::new(42));
        assert_ne!(ManagedRc::new(42), ManagedRc::new(99));
    }

    #[test]
    fn partial_eq_different_allocations() {
        let a = ManagedRc::new(42);
        let b = ManagedRc::new(42);
        assert!(!ManagedRc::ptr_eq(&a, &b));
        assert_eq!(a, b);
    }

    #[test]
    fn partial_ord_ordering() {
        assert!(ManagedRc::new(1) < ManagedRc::new(2));
        assert!(ManagedRc::new(2) > ManagedRc::new(1));
        assert!(ManagedRc::new(1) <= ManagedRc::new(1));
    }

    #[test]
    fn ptr_eq_same_allocation() {
        let a = ManagedRc::new(42);
        let b = a.clone();
        assert!(ManagedRc::ptr_eq(&a, &b));
    }

    #[test]
    fn ptr_eq_different_allocations() {
        let a = ManagedRc::new(42);
        let b = ManagedRc::new(42);
        assert!(!ManagedRc::ptr_eq(&a, &b));
    }

    #[test]
    fn zero_sized_type() {
        let a = ManagedRc::new(());
        let b = a.clone();
        assert_eq!(a.strong_count(), 2);
        drop(b);
        assert_eq!(a.strong_count(), 1);
        drop(a);
    }

    #[test]
    fn nested_managed_rc() {
        let inner = ManagedRc::new(42);
        let outer = ManagedRc::new(inner.clone());
        assert_eq!(inner.strong_count(), 2);
        assert_eq!(outer.strong_count(), 1);
        drop(outer);
        assert_eq!(inner.strong_count(), 1);
    }

    #[test]
    fn clone_then_make_mut_then_clone_again() {
        let mut a = ManagedRc::new(vec![1]);
        let b = a.clone();
        assert_eq!(a.strong_count(), 2);
        a.make_mut().push(2);
        assert!(!ManagedRc::ptr_eq(&a, &b));
        assert_eq!(a.strong_count(), 1);
        let c = a.clone();
        assert_eq!(a.strong_count(), 2);
        assert!(ManagedRc::ptr_eq(&a, &c));
        assert_eq!(*a, vec![1, 2]);
        assert_eq!(*b, vec![1]);
        assert_eq!(*c, vec![1, 2]);
    }

    #[test]
    fn clone_near_max_does_not_corrupt() {
        let rc = ManagedRc::new(42);
        unsafe { rc.ptr.as_ref() }.header.strong.set(u32::MAX - 1);
        let clone = rc.clone();
        assert_eq!(rc.strong_count(), u32::MAX);
        drop(clone);
        assert_eq!(rc.strong_count(), u32::MAX - 1);

        unsafe { rc.ptr.as_ref() }.header.strong.set(1);
    }

    #[test]
    fn make_mut_no_aliasing_violation() {
        let mut a = ManagedRc::new(vec![1, 2, 3]);
        let b = a.clone();
        let data = a.make_mut();
        data.push(4);
        data[0] = 10;
        assert_eq!(*a, vec![10, 2, 3, 4]);
        assert_eq!(*b, vec![1, 2, 3]);
    }

    #[test]
    fn make_mut_unique_no_aliasing_violation() {
        let mut rc = ManagedRc::new(vec![1]);
        let data = rc.make_mut();
        data.push(2);
        data.push(3);
        assert_eq!(*rc, vec![1, 2, 3]);
    }

    #[test]
    fn drop_runs_inner_destructor() {
        let dropped = Cell::new(false);
        let inner = ManagedRc::new(DropTracker { dropped: &dropped });
        let outer = ManagedRc::new(inner);
        drop(outer);
        assert!(dropped.get());
    }

    #[test]
    fn forget_does_not_free() {
        let dropped = Cell::new(false);
        let rc = ManagedRc::new(DropTracker { dropped: &dropped });
        let clone = rc.clone();
        std::mem::forget(clone);
        assert!(!dropped.get());
        assert_eq!(rc.strong_count(), 2);
        unsafe { rc.ptr.as_ref() }.header.strong.set(1);
    }
}
