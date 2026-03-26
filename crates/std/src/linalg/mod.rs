use anvyx_lang::provider;

use super::StdModule;

mod double_vecs;
mod float_vecs;
mod int_vecs;

use double_vecs::*;
use float_vecs::*;
use int_vecs::*;

provider!(types: [Vec2, Vec3, Vec4, Mat4, Quat, Mat3, IVec2, IVec3, IVec4, DVec2, DVec3, DVec4]);

pub fn module() -> StdModule {
    StdModule {
        name: "linalg",
        anv_source: "",
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
    }
}
