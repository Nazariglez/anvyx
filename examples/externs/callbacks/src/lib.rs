use anvyx_lang::{
    export_fn, export_methods, export_type, AnvyxConvert, AnvyxFn, ExternHandle, RuntimeError,
    Value,
};

#[export_fn]
pub fn apply(value: i64, cb: AnvyxFn<(i64,), i64>) -> Result<i64, RuntimeError> {
    cb.call(value)
}

#[export_fn]
pub fn apply_float(value: f32, cb: AnvyxFn<(f32,), f32>) -> Result<f32, RuntimeError> {
    cb.call(value)
}

#[export_fn]
pub fn each(from: i64, to: i64, cb: AnvyxFn<(i64,), ()>) -> Result<(), RuntimeError> {
    for i in from..to {
        cb.call(i)?;
    }
    Ok(())
}

#[export_fn]
pub fn double_it(x: i64) -> i64 {
    x * 2
}

#[derive(Clone)]
#[export_type]
pub struct Counter {
    #[field]
    pub value: i64,
}

#[export_methods]
impl Counter {
    #[init]
    pub fn init(value: i64) -> Self {
        Counter { value }
    }

    pub fn doubled(&self) -> i64 {
        self.value * 2
    }
}

#[export_fn]
pub fn with_counter(
    initial: i64,
    cb: AnvyxFn<(ExternHandle<Counter>,), i64>,
) -> Result<i64, RuntimeError> {
    let counter = ExternHandle::new(Counter { value: initial });
    cb.call(counter)
}

#[export_fn(ret = "[int]")]
pub fn map_range(from: i64, to: i64, cb: AnvyxFn<(i64,), i64>) -> Result<Vec<Value>, RuntimeError> {
    let mut results = vec![];
    for i in from..to {
        results.push(cb.call(i)?.into_anvyx());
    }
    Ok(results)
}

/// Takes an Anvyx-defined struct (carried as Value) and a callback that transforms it.
/// The params annotation tells the Anvyx typechecker the concrete struct type.
#[export_fn(params(point = "Point2D", cb = "fn(Point2D) -> float"), ret = "float")]
pub fn transform_point(
    point: Value,
    cb: AnvyxFn<(Value,), f32>,
) -> Result<f32, RuntimeError> {
    cb.call(point)
}

anvyx_lang::provider!(types: [Counter], apply, apply_float, each, double_it, with_counter, map_range, transform_point);
