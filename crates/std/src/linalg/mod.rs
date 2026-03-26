use anvyx_lang::provider;

use super::StdModule;

mod dvecx;
mod ivecx;
mod vecx;

use dvecx::*;
use ivecx::*;
use vecx::*;

provider!(types: [Vec2, Vec3, Vec4, Mat4, Quat, Mat3, IVec2, IVec3, IVec4, DVec2, DVec3, DVec4]);

pub fn module() -> StdModule {
    StdModule {
        name: "linalg",
        anv_source: include_str!("../linalg.anv"),
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
    }
}
