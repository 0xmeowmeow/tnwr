// Flat coloured rectangles in pixel space.
//
// glyphon draws glyphs and nothing else, so token backgrounds need their own
// pass. Everything is instanced: one instance per rectangle, corners generated
// from the vertex index, so there is no vertex buffer to keep in sync.

struct Uniforms {
    screen: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct Instance {
    @location(0) rect: vec4<f32>,   // x, y, w, h — pixels, origin top-left
    @location(1) color: vec4<f32>,  // straight alpha
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, inst: Instance) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vi];
    let px = inst.rect.xy + corner * inst.rect.zw;

    // pixels (y down) -> clip space (y up)
    let ndc = vec2<f32>(
        px.x / u.screen.x * 2.0 - 1.0,
        1.0 - px.y / u.screen.y * 2.0,
    );

    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
