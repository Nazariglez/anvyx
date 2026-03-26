use std::fmt;

use anvyx_lang::{export_methods, export_type};

#[export_type]
pub struct IVec2(pub glam::I64Vec2);

impl fmt::Display for IVec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IVec2({}, {})", self.0.x, self.0.y)
    }
}

#[export_methods]
impl IVec2 {
    #[init]
    pub fn new(x: i64, y: i64) -> Self {
        Self(glam::I64Vec2::new(x, y))
    }

    pub fn zero() -> Self {
        Self(glam::I64Vec2::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::I64Vec2::ONE)
    }

    pub fn splat(v: i64) -> Self {
        Self(glam::I64Vec2::splat(v))
    }

    #[getter]
    pub fn x(&self) -> i64 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: i64) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> i64 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: i64) {
        self.0.y = v;
    }

    pub fn abs(&self) -> IVec2 {
        IVec2(self.0.abs())
    }

    pub fn min(&self, other: &IVec2) -> IVec2 {
        IVec2(self.0.min(other.0))
    }

    pub fn max(&self, other: &IVec2) -> IVec2 {
        IVec2(self.0.max(other.0))
    }

    pub fn dot(&self, other: &IVec2) -> i64 {
        self.0.dot(other.0)
    }

    pub fn length_squared(&self) -> i64 {
        self.0.length_squared()
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &IVec2) -> IVec2 {
        IVec2(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &IVec2) -> IVec2 {
        IVec2(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &IVec2) -> IVec2 {
        IVec2(self.0 * other.0)
    }

    #[op(Self * int)]
    pub fn mul_scalar(&self, s: i64) -> IVec2 {
        IVec2(self.0 * s)
    }

    #[op(int * Self)]
    pub fn scalar_mul(&self, s: i64) -> IVec2 {
        IVec2(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &IVec2) -> IVec2 {
        IVec2(self.0 / other.0)
    }

    #[op(Self / int)]
    pub fn div_scalar(&self, s: i64) -> IVec2 {
        IVec2(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> IVec2 {
        IVec2(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &IVec2) -> bool {
        self.0 == other.0
    }
}

#[export_type]
pub struct IVec3(pub glam::I64Vec3);

impl fmt::Display for IVec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IVec3({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

#[export_methods]
impl IVec3 {
    #[init]
    pub fn new(x: i64, y: i64, z: i64) -> Self {
        Self(glam::I64Vec3::new(x, y, z))
    }

    pub fn zero() -> Self {
        Self(glam::I64Vec3::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::I64Vec3::ONE)
    }

    pub fn splat(v: i64) -> Self {
        Self(glam::I64Vec3::splat(v))
    }

    #[getter]
    pub fn x(&self) -> i64 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: i64) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> i64 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: i64) {
        self.0.y = v;
    }

    #[getter]
    pub fn z(&self) -> i64 {
        self.0.z
    }

    #[setter]
    pub fn set_z(&mut self, v: i64) {
        self.0.z = v;
    }

    pub fn abs(&self) -> IVec3 {
        IVec3(self.0.abs())
    }

    pub fn min(&self, other: &IVec3) -> IVec3 {
        IVec3(self.0.min(other.0))
    }

    pub fn max(&self, other: &IVec3) -> IVec3 {
        IVec3(self.0.max(other.0))
    }

    pub fn dot(&self, other: &IVec3) -> i64 {
        self.0.dot(other.0)
    }

    pub fn length_squared(&self) -> i64 {
        self.0.length_squared()
    }

    pub fn truncate(&self) -> IVec2 {
        IVec2(self.0.truncate())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &IVec3) -> IVec3 {
        IVec3(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &IVec3) -> IVec3 {
        IVec3(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &IVec3) -> IVec3 {
        IVec3(self.0 * other.0)
    }

    #[op(Self * int)]
    pub fn mul_scalar(&self, s: i64) -> IVec3 {
        IVec3(self.0 * s)
    }

    #[op(int * Self)]
    pub fn scalar_mul(&self, s: i64) -> IVec3 {
        IVec3(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &IVec3) -> IVec3 {
        IVec3(self.0 / other.0)
    }

    #[op(Self / int)]
    pub fn div_scalar(&self, s: i64) -> IVec3 {
        IVec3(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> IVec3 {
        IVec3(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &IVec3) -> bool {
        self.0 == other.0
    }
}

#[export_type]
pub struct IVec4(pub glam::I64Vec4);

impl fmt::Display for IVec4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IVec4({}, {}, {}, {})",
            self.0.x, self.0.y, self.0.z, self.0.w
        )
    }
}

#[export_methods]
impl IVec4 {
    #[init]
    pub fn new(x: i64, y: i64, z: i64, w: i64) -> Self {
        Self(glam::I64Vec4::new(x, y, z, w))
    }

    pub fn zero() -> Self {
        Self(glam::I64Vec4::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::I64Vec4::ONE)
    }

    pub fn splat(v: i64) -> Self {
        Self(glam::I64Vec4::splat(v))
    }

    #[getter]
    pub fn x(&self) -> i64 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: i64) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> i64 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: i64) {
        self.0.y = v;
    }

    #[getter]
    pub fn z(&self) -> i64 {
        self.0.z
    }

    #[setter]
    pub fn set_z(&mut self, v: i64) {
        self.0.z = v;
    }

    #[getter]
    pub fn w(&self) -> i64 {
        self.0.w
    }

    #[setter]
    pub fn set_w(&mut self, v: i64) {
        self.0.w = v;
    }

    pub fn abs(&self) -> IVec4 {
        IVec4(self.0.abs())
    }

    pub fn min(&self, other: &IVec4) -> IVec4 {
        IVec4(self.0.min(other.0))
    }

    pub fn max(&self, other: &IVec4) -> IVec4 {
        IVec4(self.0.max(other.0))
    }

    pub fn dot(&self, other: &IVec4) -> i64 {
        self.0.dot(other.0)
    }

    pub fn length_squared(&self) -> i64 {
        self.0.length_squared()
    }

    pub fn truncate(&self) -> IVec3 {
        IVec3(self.0.truncate())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &IVec4) -> IVec4 {
        IVec4(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &IVec4) -> IVec4 {
        IVec4(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &IVec4) -> IVec4 {
        IVec4(self.0 * other.0)
    }

    #[op(Self * int)]
    pub fn mul_scalar(&self, s: i64) -> IVec4 {
        IVec4(self.0 * s)
    }

    #[op(int * Self)]
    pub fn scalar_mul(&self, s: i64) -> IVec4 {
        IVec4(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &IVec4) -> IVec4 {
        IVec4(self.0 / other.0)
    }

    #[op(Self / int)]
    pub fn div_scalar(&self, s: i64) -> IVec4 {
        IVec4(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> IVec4 {
        IVec4(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &IVec4) -> bool {
        self.0 == other.0
    }
}
