use std::fmt;

use anvyx_lang::{export_methods, export_type};

fn fmt_double(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

#[derive(Clone)]
#[export_type]
pub struct DVec2(pub glam::DVec2);

impl fmt::Display for DVec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DVec2({}, {})",
            fmt_double(self.0.x),
            fmt_double(self.0.y)
        )
    }
}

#[export_methods]
impl DVec2 {
    #[init]
    pub fn new(x: f64, y: f64) -> Self {
        Self(glam::DVec2::new(x, y))
    }

    pub fn zero() -> Self {
        Self(glam::DVec2::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::DVec2::ONE)
    }

    pub fn splat(v: f64) -> Self {
        Self(glam::DVec2::splat(v))
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

    pub fn length_squared(&self) -> f64 {
        self.0.length_squared()
    }

    pub fn dot(&self, other: &DVec2) -> f64 {
        self.0.dot(other.0)
    }

    pub fn normalize(&self) -> DVec2 {
        DVec2(self.0.normalize())
    }

    pub fn normalize_or_zero(&self) -> DVec2 {
        DVec2(self.0.normalize_or_zero())
    }

    pub fn distance(&self, other: &DVec2) -> f64 {
        self.0.distance(other.0)
    }

    pub fn lerp(&self, other: &DVec2, t: f64) -> DVec2 {
        DVec2(self.0.lerp(other.0, t))
    }

    pub fn abs(&self) -> DVec2 {
        DVec2(self.0.abs())
    }

    pub fn min(&self, other: &DVec2) -> DVec2 {
        DVec2(self.0.min(other.0))
    }

    pub fn max(&self, other: &DVec2) -> DVec2 {
        DVec2(self.0.max(other.0))
    }

    pub fn clamp(&self, min: &DVec2, max: &DVec2) -> DVec2 {
        DVec2(self.0.clamp(min.0, max.0))
    }

    pub fn perp(&self) -> DVec2 {
        DVec2(self.0.perp())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &DVec2) -> DVec2 {
        DVec2(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &DVec2) -> DVec2 {
        DVec2(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &DVec2) -> DVec2 {
        DVec2(self.0 * other.0)
    }

    #[op(Self * double)]
    pub fn mul_scalar(&self, s: f64) -> DVec2 {
        DVec2(self.0 * s)
    }

    #[op(double * Self)]
    pub fn scalar_mul(&self, s: f64) -> DVec2 {
        DVec2(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &DVec2) -> DVec2 {
        DVec2(self.0 / other.0)
    }

    #[op(Self / double)]
    pub fn div_scalar(&self, s: f64) -> DVec2 {
        DVec2(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> DVec2 {
        DVec2(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &DVec2) -> bool {
        self.0 == other.0
    }
}

#[derive(Clone)]
#[export_type]
pub struct DVec3(pub glam::DVec3);

impl fmt::Display for DVec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DVec3({}, {}, {})",
            fmt_double(self.0.x),
            fmt_double(self.0.y),
            fmt_double(self.0.z)
        )
    }
}

#[export_methods]
impl DVec3 {
    #[init]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self(glam::DVec3::new(x, y, z))
    }

    pub fn zero() -> Self {
        Self(glam::DVec3::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::DVec3::ONE)
    }

    pub fn splat(v: f64) -> Self {
        Self(glam::DVec3::splat(v))
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

    #[getter]
    pub fn z(&self) -> f64 {
        self.0.z
    }

    #[setter]
    pub fn set_z(&mut self, v: f64) {
        self.0.z = v;
    }

    pub fn length(&self) -> f64 {
        self.0.length()
    }

    pub fn length_squared(&self) -> f64 {
        self.0.length_squared()
    }

    pub fn dot(&self, other: &DVec3) -> f64 {
        self.0.dot(other.0)
    }

    pub fn cross(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0.cross(other.0))
    }

    pub fn normalize(&self) -> DVec3 {
        DVec3(self.0.normalize())
    }

    pub fn normalize_or_zero(&self) -> DVec3 {
        DVec3(self.0.normalize_or_zero())
    }

    pub fn distance(&self, other: &DVec3) -> f64 {
        self.0.distance(other.0)
    }

    pub fn lerp(&self, other: &DVec3, t: f64) -> DVec3 {
        DVec3(self.0.lerp(other.0, t))
    }

    pub fn abs(&self) -> DVec3 {
        DVec3(self.0.abs())
    }

    pub fn min(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0.min(other.0))
    }

    pub fn max(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0.max(other.0))
    }

    pub fn clamp(&self, min: &DVec3, max: &DVec3) -> DVec3 {
        DVec3(self.0.clamp(min.0, max.0))
    }

    pub fn truncate(&self) -> DVec2 {
        DVec2(self.0.truncate())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0 * other.0)
    }

    #[op(Self * double)]
    pub fn mul_scalar(&self, s: f64) -> DVec3 {
        DVec3(self.0 * s)
    }

    #[op(double * Self)]
    pub fn scalar_mul(&self, s: f64) -> DVec3 {
        DVec3(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &DVec3) -> DVec3 {
        DVec3(self.0 / other.0)
    }

    #[op(Self / double)]
    pub fn div_scalar(&self, s: f64) -> DVec3 {
        DVec3(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> DVec3 {
        DVec3(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &DVec3) -> bool {
        self.0 == other.0
    }
}

#[derive(Clone)]
#[export_type]
pub struct DVec4(pub glam::DVec4);

impl fmt::Display for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DVec4({}, {}, {}, {})",
            fmt_double(self.0.x),
            fmt_double(self.0.y),
            fmt_double(self.0.z),
            fmt_double(self.0.w)
        )
    }
}

#[export_methods]
impl DVec4 {
    #[init]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self(glam::DVec4::new(x, y, z, w))
    }

    pub fn zero() -> Self {
        Self(glam::DVec4::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::DVec4::ONE)
    }

    pub fn splat(v: f64) -> Self {
        Self(glam::DVec4::splat(v))
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

    #[getter]
    pub fn z(&self) -> f64 {
        self.0.z
    }

    #[setter]
    pub fn set_z(&mut self, v: f64) {
        self.0.z = v;
    }

    #[getter]
    pub fn w(&self) -> f64 {
        self.0.w
    }

    #[setter]
    pub fn set_w(&mut self, v: f64) {
        self.0.w = v;
    }

    pub fn length(&self) -> f64 {
        self.0.length()
    }

    pub fn dot(&self, other: &DVec4) -> f64 {
        self.0.dot(other.0)
    }

    pub fn normalize(&self) -> DVec4 {
        DVec4(self.0.normalize())
    }

    pub fn normalize_or_zero(&self) -> DVec4 {
        DVec4(self.0.normalize_or_zero())
    }

    pub fn lerp(&self, other: &DVec4, t: f64) -> DVec4 {
        DVec4(self.0.lerp(other.0, t))
    }

    pub fn abs(&self) -> DVec4 {
        DVec4(self.0.abs())
    }

    pub fn min(&self, other: &DVec4) -> DVec4 {
        DVec4(self.0.min(other.0))
    }

    pub fn max(&self, other: &DVec4) -> DVec4 {
        DVec4(self.0.max(other.0))
    }

    pub fn truncate(&self) -> DVec3 {
        DVec3(self.0.truncate())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &DVec4) -> DVec4 {
        DVec4(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &DVec4) -> DVec4 {
        DVec4(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &DVec4) -> DVec4 {
        DVec4(self.0 * other.0)
    }

    #[op(Self * double)]
    pub fn mul_scalar(&self, s: f64) -> DVec4 {
        DVec4(self.0 * s)
    }

    #[op(double * Self)]
    pub fn scalar_mul(&self, s: f64) -> DVec4 {
        DVec4(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &DVec4) -> DVec4 {
        DVec4(self.0 / other.0)
    }

    #[op(Self / double)]
    pub fn div_scalar(&self, s: f64) -> DVec4 {
        DVec4(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> DVec4 {
        DVec4(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &DVec4) -> bool {
        self.0 == other.0
    }
}
