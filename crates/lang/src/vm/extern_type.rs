use super::handle_store::HandleStore;
use super::managed_rc::ManagedRc;
use super::value::{ExternHandleData, Value};
use std::cell::RefCell;

pub trait AnvyxExternType: Sized + 'static {
    const TYPE_NAME: &'static str;
    fn with_store<R>(f: impl FnOnce(&RefCell<HandleStore<Self>>) -> R) -> R;
    fn cleanup(id: u64);
    fn to_display(id: u64) -> String;
}

pub fn extern_handle<T: AnvyxExternType>(value: T) -> Value {
    let id = T::with_store(|s| s.borrow_mut().insert(value));
    Value::ExternHandle(ManagedRc::new(ExternHandleData {
        id,
        drop_fn: T::cleanup,
        type_name: T::TYPE_NAME,
        to_string_fn: T::to_display,
    }))
}
