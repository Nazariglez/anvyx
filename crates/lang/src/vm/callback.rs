use std::{cell::RefCell, marker::PhantomData};

use super::{
    extern_type::AnvyxConvert,
    runtime::{ExternRegistry, VM},
    value::{RuntimeError, Value},
};

pub trait VmContext {
    fn call_closure(&mut self, closure: &Value, args: Vec<Value>) -> Result<Value, RuntimeError>;
}

pub(crate) struct VmWithRegistry<'vm, 'prog, 'reg> {
    pub(crate) vm: &'vm mut VM<'prog>,
    pub(crate) registry: &'reg ExternRegistry,
}

impl VmContext for VmWithRegistry<'_, '_, '_> {
    fn call_closure(&mut self, closure: &Value, args: Vec<Value>) -> Result<Value, RuntimeError> {
        self.vm.call_closure(self.registry, closure, args)
    }
}

// Stack of raw pointers to VmWithRegistry locals,
// each entry is pushed by callback_scope and popped by CallbackScopeGuard::drop
thread_local! {
    static CALLBACK_CTX: RefCell<Vec<*mut (dyn VmContext + 'static)>> = const { RefCell::new(vec![]) };
}

pub(crate) struct CallbackScopeGuard;

impl Drop for CallbackScopeGuard {
    fn drop(&mut self) {
        CALLBACK_CTX.with(|c| {
            c.borrow_mut().pop();
        });
    }
}

pub(crate) fn callback_scope(ctx: &mut dyn VmContext) -> CallbackScopeGuard {
    // SAFETY: transmute only erases the lifetime on a fat pointer (data + vtable unchanged)
    // CallbackScopeGuard::drop pops this pointer before the referent goes out of scope
    // clippy suggests "as" cast, but that cannot erase the borrow lifetime on a trait object pointer
    #[allow(clippy::transmute_ptr_to_ptr)]
    let ptr: *mut (dyn VmContext + 'static) =
        unsafe { std::mem::transmute::<*mut dyn VmContext, *mut (dyn VmContext + 'static)>(ctx) };
    CALLBACK_CTX.with(|c| c.borrow_mut().push(ptr));
    CallbackScopeGuard
}

// Pushes the pointer back when with_callback_ctx finishes (or panics)
struct RestoreGuard(*mut (dyn VmContext + 'static));

impl Drop for RestoreGuard {
    fn drop(&mut self) {
        CALLBACK_CTX.with(|c| c.borrow_mut().push(self.0));
    }
}

pub fn with_callback_ctx<R>(
    f: impl FnOnce(&mut dyn VmContext) -> Result<R, RuntimeError>,
) -> Result<R, RuntimeError> {
    // pop the pointer so nested calls at the same depth can't alias it
    // reentrant callbacks (via call_closure -> run_until_depth -> Op::CallExtern)
    // push a new entry, so they work fine on the now shorter stack
    let ptr = CALLBACK_CTX.with(|c| {
        c.borrow_mut()
            .pop()
            .ok_or_else(|| RuntimeError::new("callback invoked outside of an active VM handler"))
    })?;

    // RestoreGuard pushes the pointer back when f returns (or panics),
    // so CallbackScopeGuard::drop can still pop it
    let _restore = RestoreGuard(ptr);

    // SAFETY: the pointer is valid because callback_scope created it from a local
    // VmWithRegistry that outlives the handler call. Exclusive access is guaranteed
    // because we just popped the pointer, no one else can reach it.
    let ctx = unsafe { &mut *ptr };
    f(ctx)
}

/// A typed wrapper around an Anvyx closure value
///
/// `A` is a tuple of argument types; `R` is the return type
/// Both must implement [`AnvyxConvert`]
#[derive(Clone)]
pub struct AnvyxFn<A, R> {
    value: Value,
    _phantom: PhantomData<fn(A) -> R>,
}

impl<A, R> AnvyxConvert for AnvyxFn<A, R> {
    fn anvyx_type() -> &'static str {
        "<callback>"
    }
    fn anvyx_option_type() -> &'static str {
        "<callback>?"
    }

    fn into_anvyx(self) -> Value {
        self.value
    }

    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        match v {
            Value::Closure(_) => Ok(AnvyxFn {
                value: v.clone(),
                _phantom: PhantomData,
            }),
            _ => Err(RuntimeError::new("expected callable")),
        }
    }
}

macro_rules! impl_anvyx_fn_call {
    () => {
        impl<R: AnvyxConvert> AnvyxFn<(), R> {
            pub fn call(&self) -> Result<R, RuntimeError> {
                with_callback_ctx(|ctx| {
                    let raw = ctx.call_closure(&self.value, vec![])?;
                    R::from_anvyx(&raw)
                })
            }
        }
    };
    ($($name:ident : $T:ident),+) => {
        impl<$($T: AnvyxConvert,)+ R: AnvyxConvert> AnvyxFn<($($T,)+), R> {
            #[allow(clippy::too_many_arguments)]
            pub fn call(&self, $($name: $T),+) -> Result<R, RuntimeError> {
                with_callback_ctx(|ctx| {
                    let raw = ctx.call_closure(
                        &self.value,
                        vec![$($name.into_anvyx()),+],
                    )?;
                    R::from_anvyx(&raw)
                })
            }
        }
    };
}

// support calling shared Anvyx callbacks from Rust with up to 9 arguments
impl_anvyx_fn_call!();
impl_anvyx_fn_call!(a0: T0);
impl_anvyx_fn_call!(a0: T0, a1: T1);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2, a3: T3);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2, a3: T3, a4: T4);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2, a3: T3, a4: T4, a5: T5);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2, a3: T3, a4: T4, a5: T5, a6: T6);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2, a3: T3, a4: T4, a5: T5, a6: T6, a7: T7);
impl_anvyx_fn_call!(a0: T0, a1: T1, a2: T2, a3: T3, a4: T4, a5: T5, a6: T6, a7: T7, a8: T8);
