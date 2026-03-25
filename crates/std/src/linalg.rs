use std::fmt;

use anvyx_lang::{export_methods, export_type, provider};

use super::StdModule;

#[export_type]
pub struct Vec2(pub glam::DVec2);

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn fmt_float(v: f64) -> String {
            if v.fract() == 0.0 && v.is_finite() {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        write!(f, "Vec2({}, {})", fmt_float(self.0.x), fmt_float(self.0.y))
    }
}

#[export_methods]
impl Vec2 {
    #[init]
    pub fn new(x: f64, y: f64) -> Self {
        Self(glam::DVec2::new(x, y))
    }

    pub fn zero() -> Self {
        Self(glam::DVec2::ZERO)
    }

    #[getter]
    pub fn x(&self) -> f64 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: f64) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> f64 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: f64) {
        self.0.y = v;
    }

    pub fn length(&self) -> f64 {
        self.0.length()
    }

    pub fn dot(&self, other: &Vec2) -> f64 {
        self.0.dot(other.0)
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 * other.0)
    }

    #[op(Self * float)]
    pub fn mul_scalar(&self, s: f64) -> Vec2 {
        Vec2(self.0 * s)
    }

    #[op(float * Self)]
    pub fn scalar_mul(&self, s: f64) -> Vec2 {
        Vec2(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 / other.0)
    }

    #[op(Self / float)]
    pub fn div_scalar(&self, s: f64) -> Vec2 {
        Vec2(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> Vec2 {
        Vec2(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &Vec2) -> bool {
        self.0 == other.0
    }
}

provider!(types: [Vec2]);

pub fn module() -> StdModule {
    StdModule {
        name: "linalg",
        anv_source: "",
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
    }
}
