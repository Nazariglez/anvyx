use std::collections::HashMap;
use super::value::RuntimeError;

pub struct HandleStore<T> {
    map: HashMap<u64, T>,
    next_id: u64,
}

impl<T> HandleStore<T> {
    pub fn new() -> Self {
        Self { map: HashMap::new(), next_id: 0 }
    }

    pub fn insert(&mut self, value: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(id, value);
        id
    }

    pub fn get(&self, id: u64) -> Result<&T, RuntimeError> {
        self.map.get(&id).ok_or_else(|| RuntimeError::new(format!("invalid extern handle: {id}")))
    }

    pub fn get_mut(&mut self, id: u64) -> Result<&mut T, RuntimeError> {
        self.map.get_mut(&id).ok_or_else(|| RuntimeError::new(format!("invalid extern handle: {id}")))
    }

    pub fn remove(&mut self, id: u64) -> Result<T, RuntimeError> {
        self.map.remove(&id).ok_or_else(|| RuntimeError::new(format!("invalid extern handle: {id}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut store = HandleStore::new();
        let id = store.insert(42);
        assert_eq!(*store.get(id).unwrap(), 42);
    }

    #[test]
    fn insert_and_get_mut() {
        let mut store = HandleStore::new();
        let id = store.insert(10);
        *store.get_mut(id).unwrap() = 20;
        assert_eq!(*store.get(id).unwrap(), 20);
    }

    #[test]
    fn insert_and_remove() {
        let mut store = HandleStore::new();
        let id = store.insert("hello".to_string());
        let val = store.remove(id).unwrap();
        assert_eq!(val, "hello");
        assert!(store.get(id).is_err());
    }

    #[test]
    fn missing_key_errors() {
        let store: HandleStore<i32> = HandleStore::new();
        let err = store.get(999).unwrap_err();
        assert!(err.message.contains("invalid extern handle"));

        let mut store2: HandleStore<i32> = HandleStore::new();
        assert!(store2.get_mut(999).is_err());
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
}
