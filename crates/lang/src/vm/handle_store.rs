use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
};

use super::value::RuntimeError;

pub struct HandleStore<T> {
    entries: HashMap<u64, RefCell<T>>,
    next_id: u64,
}

impl<T> Default for HandleStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HandleStore<T> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn insert(&mut self, value: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, RefCell::new(value));
        id
    }

    pub fn borrow(&self, id: u64) -> Result<Ref<'_, T>, RuntimeError> {
        self.entries
            .get(&id)
            .ok_or_else(|| RuntimeError::new(format!("invalid extern handle: {id}")))?
            .try_borrow()
            .map_err(|_| {
                RuntimeError::new(format!("extern handle {id} is already mutably borrowed"))
            })
    }

    pub fn borrow_mut(&self, id: u64) -> Result<RefMut<'_, T>, RuntimeError> {
        self.entries
            .get(&id)
            .ok_or_else(|| RuntimeError::new(format!("invalid extern handle: {id}")))?
            .try_borrow_mut()
            .map_err(|_| RuntimeError::new(format!("extern handle {id} is already borrowed")))
    }

    pub fn remove(&mut self, id: u64) -> Result<T, RuntimeError> {
        self.entries
            .remove(&id)
            .map(RefCell::into_inner)
            .ok_or_else(|| RuntimeError::new(format!("invalid extern handle: {id}")))
    }

    pub fn clone_value(&self, id: u64) -> Result<T, RuntimeError>
    where
        T: Clone,
    {
        let guard = self.borrow(id)?;
        Ok((*guard).clone())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut store = HandleStore::new();
        let id = store.insert(42);
        assert_eq!(*store.borrow(id).unwrap(), 42);
    }

    #[test]
    fn insert_and_get_mut() {
        let mut store = HandleStore::new();
        let id = store.insert(10);
        *store.borrow_mut(id).unwrap() = 20;
        assert_eq!(*store.borrow(id).unwrap(), 20);
    }

    #[test]
    fn insert_and_remove() {
        let mut store = HandleStore::new();
        let id = store.insert("hello".to_string());
        let val = store.remove(id).unwrap();
        assert_eq!(val, "hello");
        assert!(store.borrow(id).is_err());
    }

    #[test]
    fn missing_key_errors() {
        let store: HandleStore<i32> = HandleStore::new();
        let err = store.borrow(999).unwrap_err();
        assert!(err.message.contains("invalid extern handle"));

        let mut store2: HandleStore<i32> = HandleStore::new();
        assert!(store2.borrow_mut(999).is_err());
        assert!(store2.remove(999).is_err());
    }

    #[test]
    fn sequential_ids() {
        let mut store = HandleStore::new();
        let a = store.insert("a");
        let b = store.insert("b");
        let c = store.insert("c");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
    }

    #[test]
    fn concurrent_immutable_borrows() {
        let mut store = HandleStore::new();
        let a = store.insert(10);
        let b = store.insert(20);
        let ref_a = store.borrow(a).unwrap();
        let ref_b = store.borrow(b).unwrap();
        assert_eq!(*ref_a, 10);
        assert_eq!(*ref_b, 20);
    }

    #[test]
    fn mut_and_immut_different_entries() {
        let mut store = HandleStore::new();
        let a = store.insert(10);
        let b = store.insert(20);
        let mut ref_a = store.borrow_mut(a).unwrap();
        let ref_b = store.borrow(b).unwrap();
        *ref_a = 30;
        assert_eq!(*ref_b, 20);
        drop(ref_a);
        assert_eq!(*store.borrow(a).unwrap(), 30);
    }

    #[test]
    fn same_entry_double_borrow_fails() {
        let mut store = HandleStore::new();
        let id = store.insert(42);
        let _ref_mut = store.borrow_mut(id).unwrap();
        let result = store.borrow(id);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("already mutably borrowed"));
    }

    #[test]
    fn clone_value_without_removing() {
        let mut store = HandleStore::new();
        let id = store.insert("hello".to_string());

        let a = store.clone_value(id).unwrap();
        assert_eq!(a, "hello");

        // second clone succeeds, entry was not removed
        let b = store.clone_value(id).unwrap();
        assert_eq!(b, "hello");

        assert_eq!(*store.borrow(id).unwrap(), "hello");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn clone_value_missing_id() {
        let store: HandleStore<String> = HandleStore::new();
        assert!(store.clone_value(999).is_err());
    }

    #[test]
    fn store_len_tracks_entries() {
        let mut store = HandleStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        let a = store.insert(1);
        let b = store.insert(2);
        let c = store.insert(3);
        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
        store.remove(b).unwrap();
        assert_eq!(store.len(), 2);
        store.remove(a).unwrap();
        store.remove(c).unwrap();
        assert!(store.is_empty());
    }
}
