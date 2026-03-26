use std::fmt;

use anvyx_lang::{export_methods, export_type};

fn fmt_float(v: f32) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

#[export_type]
pub struct Vec2(pub glam::Vec2);

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vec2({}, {})", fmt_float(self.0.x), fmt_float(self.0.y))
    }
}

#[export_methods]
impl Vec2 {
    #[init]
    pub fn new(x: f32, y: f32) -> Self {
        Self(glam::Vec2::new(x, y))
    }

    pub fn zero() -> Self {
        Self(glam::Vec2::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::Vec2::ONE)
    }

    pub fn splat(v: f32) -> Self {
        Self(glam::Vec2::splat(v))
    }

    #[getter]
    pub fn x(&self) -> f32 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: f32) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> f32 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: f32) {
        self.0.y = v;
    }

    pub fn length(&self) -> f32 {
        self.0.length()
    }

    pub fn length_squared(&self) -> f32 {
        self.0.length_squared()
    }

    pub fn dot(&self, other: &Vec2) -> f32 {
        self.0.dot(other.0)
    }

    pub fn normalize(&self) -> Vec2 {
        Vec2(self.0.normalize())
    }

    pub fn normalize_or_zero(&self) -> Vec2 {
        Vec2(self.0.normalize_or_zero())
    }

    pub fn distance(&self, other: &Vec2) -> f32 {
        self.0.distance(other.0)
    }

    pub fn lerp(&self, other: &Vec2, t: f32) -> Vec2 {
        Vec2(self.0.lerp(other.0, t))
    }

    pub fn abs(&self) -> Vec2 {
        Vec2(self.0.abs())
    }

    pub fn min(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0.min(other.0))
    }

    pub fn max(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0.max(other.0))
    }

    pub fn clamp(&self, min: &Vec2, max: &Vec2) -> Vec2 {
        Vec2(self.0.clamp(min.0, max.0))
    }

    pub fn perp(&self) -> Vec2 {
        Vec2(self.0.perp())
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
    pub fn mul_scalar(&self, s: f32) -> Vec2 {
        Vec2(self.0 * s)
    }

    #[op(float * Self)]
    pub fn scalar_mul(&self, s: f32) -> Vec2 {
        Vec2(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 / other.0)
    }

    #[op(Self / float)]
    pub fn div_scalar(&self, s: f32) -> Vec2 {
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

#[export_type]
pub struct Vec3(pub glam::Vec3);

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Vec3({}, {}, {})",
            fmt_float(self.0.x),
            fmt_float(self.0.y),
            fmt_float(self.0.z)
        )
    }
}

#[export_methods]
impl Vec3 {
    #[init]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glam::Vec3::new(x, y, z))
    }

    pub fn zero() -> Self {
        Self(glam::Vec3::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::Vec3::ONE)
    }

    pub fn splat(v: f32) -> Self {
        Self(glam::Vec3::splat(v))
    }

    #[getter]
    pub fn x(&self) -> f32 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: f32) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> f32 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: f32) {
        self.0.y = v;
    }

    #[getter]
    pub fn z(&self) -> f32 {
        self.0.z
    }

    #[setter]
    pub fn set_z(&mut self, v: f32) {
        self.0.z = v;
    }

    pub fn length(&self) -> f32 {
        self.0.length()
    }

    pub fn length_squared(&self) -> f32 {
        self.0.length_squared()
    }

    pub fn dot(&self, other: &Vec3) -> f32 {
        self.0.dot(other.0)
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0.cross(other.0))
    }

    pub fn normalize(&self) -> Vec3 {
        Vec3(self.0.normalize())
    }

    pub fn normalize_or_zero(&self) -> Vec3 {
        Vec3(self.0.normalize_or_zero())
    }

    pub fn distance(&self, other: &Vec3) -> f32 {
        self.0.distance(other.0)
    }

    pub fn lerp(&self, other: &Vec3, t: f32) -> Vec3 {
        Vec3(self.0.lerp(other.0, t))
    }

    pub fn abs(&self) -> Vec3 {
        Vec3(self.0.abs())
    }

    pub fn min(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0.min(other.0))
    }

    pub fn max(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0.max(other.0))
    }

    pub fn clamp(&self, min: &Vec3, max: &Vec3) -> Vec3 {
        Vec3(self.0.clamp(min.0, max.0))
    }

    pub fn truncate(&self) -> Vec2 {
        Vec2(self.0.truncate())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 * other.0)
    }

    #[op(Self * float)]
    pub fn mul_scalar(&self, s: f32) -> Vec3 {
        Vec3(self.0 * s)
    }

    #[op(float * Self)]
    pub fn scalar_mul(&self, s: f32) -> Vec3 {
        Vec3(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 / other.0)
    }

    #[op(Self / float)]
    pub fn div_scalar(&self, s: f32) -> Vec3 {
        Vec3(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> Vec3 {
        Vec3(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &Vec3) -> bool {
        self.0 == other.0
    }
}

#[export_type]
pub struct Vec4(pub glam::Vec4);

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Vec4({}, {}, {}, {})",
            fmt_float(self.0.x),
            fmt_float(self.0.y),
            fmt_float(self.0.z),
            fmt_float(self.0.w)
        )
    }
}

#[export_methods]
impl Vec4 {
    #[init]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(glam::Vec4::new(x, y, z, w))
    }

    pub fn zero() -> Self {
        Self(glam::Vec4::ZERO)
    }

    pub fn one() -> Self {
        Self(glam::Vec4::ONE)
    }

    pub fn splat(v: f32) -> Self {
        Self(glam::Vec4::splat(v))
    }

    #[getter]
    pub fn x(&self) -> f32 {
        self.0.x
    }

    #[setter]
    pub fn set_x(&mut self, v: f32) {
        self.0.x = v;
    }

    #[getter]
    pub fn y(&self) -> f32 {
        self.0.y
    }

    #[setter]
    pub fn set_y(&mut self, v: f32) {
        self.0.y = v;
    }

    #[getter]
    pub fn z(&self) -> f32 {
        self.0.z
    }

    #[setter]
    pub fn set_z(&mut self, v: f32) {
        self.0.z = v;
    }

    #[getter]
    pub fn w(&self) -> f32 {
        self.0.w
    }

    #[setter]
    pub fn set_w(&mut self, v: f32) {
        self.0.w = v;
    }

    pub fn length(&self) -> f32 {
        self.0.length()
    }

    pub fn dot(&self, other: &Vec4) -> f32 {
        self.0.dot(other.0)
    }

    pub fn normalize(&self) -> Vec4 {
        Vec4(self.0.normalize())
    }

    pub fn normalize_or_zero(&self) -> Vec4 {
        Vec4(self.0.normalize_or_zero())
    }

    pub fn lerp(&self, other: &Vec4, t: f32) -> Vec4 {
        Vec4(self.0.lerp(other.0, t))
    }

    pub fn abs(&self) -> Vec4 {
        Vec4(self.0.abs())
    }

    pub fn min(&self, other: &Vec4) -> Vec4 {
        Vec4(self.0.min(other.0))
    }

    pub fn max(&self, other: &Vec4) -> Vec4 {
        Vec4(self.0.max(other.0))
    }

    pub fn truncate(&self) -> Vec3 {
        Vec3(self.0.truncate())
    }

    #[op(Self + Self)]
    pub fn add(&self, other: &Vec4) -> Vec4 {
        Vec4(self.0 + other.0)
    }

    #[op(Self - Self)]
    pub fn sub(&self, other: &Vec4) -> Vec4 {
        Vec4(self.0 - other.0)
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &Vec4) -> Vec4 {
        Vec4(self.0 * other.0)
    }

    #[op(Self * float)]
    pub fn mul_scalar(&self, s: f32) -> Vec4 {
        Vec4(self.0 * s)
    }

    #[op(float * Self)]
    pub fn scalar_mul(&self, s: f32) -> Vec4 {
        Vec4(s * self.0)
    }

    #[op(Self / Self)]
    pub fn div(&self, other: &Vec4) -> Vec4 {
        Vec4(self.0 / other.0)
    }

    #[op(Self / float)]
    pub fn div_scalar(&self, s: f32) -> Vec4 {
        Vec4(self.0 / s)
    }

    #[op(-Self)]
    pub fn neg(&self) -> Vec4 {
        Vec4(-self.0)
    }

    #[op(Self == Self)]
    pub fn eq(&self, other: &Vec4) -> bool {
        self.0 == other.0
    }
}

#[export_type]
pub struct Mat4(pub glam::Mat4);

impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mat4")
    }
}

#[export_methods]
impl Mat4 {
    pub fn identity() -> Mat4 {
        Mat4(glam::Mat4::IDENTITY)
    }

    pub fn from_translation(v: &Vec3) -> Mat4 {
        Mat4(glam::Mat4::from_translation(v.0))
    }

    pub fn from_scale(v: &Vec3) -> Mat4 {
        Mat4(glam::Mat4::from_scale(v.0))
    }

    pub fn from_rotation_x(angle: f32) -> Mat4 {
        Mat4(glam::Mat4::from_rotation_x(angle))
    }

    pub fn from_rotation_y(angle: f32) -> Mat4 {
        Mat4(glam::Mat4::from_rotation_y(angle))
    }

    pub fn from_rotation_z(angle: f32) -> Mat4 {
        Mat4(glam::Mat4::from_rotation_z(angle))
    }

    pub fn look_at_rh(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat4 {
        Mat4(glam::Mat4::look_at_rh(eye.0, center.0, up.0))
    }

    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        Mat4(glam::Mat4::perspective_rh(fov_y, aspect, near, far))
    }

    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Mat4 {
        Mat4(glam::Mat4::orthographic_rh(
            left, right, bottom, top, near, far,
        ))
    }

    pub fn inverse(&self) -> Mat4 {
        Mat4(self.0.inverse())
    }

    pub fn transpose(&self) -> Mat4 {
        Mat4(self.0.transpose())
    }

    pub fn determinant(&self) -> f32 {
        self.0.determinant()
    }

    pub fn mul_vec4(&self, v: &Vec4) -> Vec4 {
        Vec4(self.0.mul_vec4(v.0))
    }

    pub fn transform_point3(&self, v: &Vec3) -> Vec3 {
        Vec3(self.0.transform_point3(v.0))
    }

    pub fn transform_vector3(&self, v: &Vec3) -> Vec3 {
        Vec3(self.0.transform_vector3(v.0))
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &Mat4) -> Mat4 {
        Mat4(self.0 * other.0)
    }

    #[op(Self * Vec4)]
    pub fn mul_vec4_op(&self, v: &Vec4) -> Vec4 {
        Vec4(self.0 * v.0)
    }
}

#[export_type]
pub struct Quat(pub glam::Quat);

impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quat({}, {}, {}, {})",
            fmt_float(self.0.x),
            fmt_float(self.0.y),
            fmt_float(self.0.z),
            fmt_float(self.0.w)
        )
    }
}

#[export_methods]
impl Quat {
    pub fn identity() -> Self {
        Self(glam::Quat::IDENTITY)
    }

    pub fn from_rotation_x(angle: f32) -> Self {
        Self(glam::Quat::from_rotation_x(angle))
    }

    pub fn from_rotation_y(angle: f32) -> Self {
        Self(glam::Quat::from_rotation_y(angle))
    }

    pub fn from_rotation_z(angle: f32) -> Self {
        Self(glam::Quat::from_rotation_z(angle))
    }

    pub fn from_axis_angle(axis: &Vec3, angle: f32) -> Self {
        Self(glam::Quat::from_axis_angle(axis.0, angle))
    }

    pub fn x(&self) -> f32 {
        self.0.x
    }

    pub fn y(&self) -> f32 {
        self.0.y
    }

    pub fn z(&self) -> f32 {
        self.0.z
    }

    pub fn w(&self) -> f32 {
        self.0.w
    }

    pub fn length(&self) -> f32 {
        self.0.length()
    }

    pub fn normalize(&self) -> Quat {
        Quat(self.0.normalize())
    }

    pub fn conjugate(&self) -> Quat {
        Quat(self.0.conjugate())
    }

    pub fn inverse(&self) -> Quat {
        Quat(self.0.inverse())
    }

    pub fn dot(&self, other: &Quat) -> f32 {
        self.0.dot(other.0)
    }

    pub fn lerp(&self, other: &Quat, t: f32) -> Quat {
        Quat(self.0.lerp(other.0, t))
    }

    pub fn slerp(&self, other: &Quat, t: f32) -> Quat {
        Quat(self.0.slerp(other.0, t))
    }

    pub fn mul_vec3(&self, v: &Vec3) -> Vec3 {
        Vec3(self.0.mul_vec3(v.0))
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &Quat) -> Quat {
        Quat(self.0 * other.0)
    }

    #[op(Self * Vec3)]
    pub fn mul_vec3_op(&self, v: &Vec3) -> Vec3 {
        Vec3(self.0 * v.0)
    }
}

#[export_type]
pub struct Mat3(pub glam::Mat3);

impl fmt::Display for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mat3")
    }
}

#[export_methods]
impl Mat3 {
    pub fn identity() -> Mat3 {
        Mat3(glam::Mat3::IDENTITY)
    }

    pub fn from_scale(v: &Vec2) -> Mat3 {
        Mat3(glam::Mat3::from_scale(v.0))
    }

    pub fn from_angle(angle: f32) -> Mat3 {
        Mat3(glam::Mat3::from_angle(angle))
    }

    pub fn from_translation(v: &Vec2) -> Mat3 {
        Mat3(glam::Mat3::from_translation(v.0))
    }

    pub fn inverse(&self) -> Mat3 {
        Mat3(self.0.inverse())
    }

    pub fn transpose(&self) -> Mat3 {
        Mat3(self.0.transpose())
    }

    pub fn determinant(&self) -> f32 {
        self.0.determinant()
    }

    pub fn mul_vec3(&self, v: &Vec3) -> Vec3 {
        Vec3(self.0.mul_vec3(v.0))
    }

    pub fn transform_point2(&self, v: &Vec2) -> Vec2 {
        Vec2(self.0.transform_point2(v.0))
    }

    pub fn transform_vector2(&self, v: &Vec2) -> Vec2 {
        Vec2(self.0.transform_vector2(v.0))
    }

    #[op(Self * Self)]
    pub fn mul(&self, other: &Mat3) -> Mat3 {
        Mat3(self.0 * other.0)
    }

    #[op(Self * Vec3)]
    pub fn mul_vec3_op(&self, v: &Vec3) -> Vec3 {
        Vec3(self.0 * v.0)
    }
}
