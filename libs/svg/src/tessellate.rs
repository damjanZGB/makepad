use crate::path::*;

// Output vertex: position + texcoord for AA + distance along stroke
#[derive(Clone, Copy, Default, Debug)]
pub struct VVertex {
    pub x: f32,
    pub y: f32,
    pub u: f32,
    pub v: f32,
    pub stroke_dist: f32,
    /// Maximum distance from this vertex to any other vertex it shares a triangle with.
    /// Used for early clip rejection in the vertex shader.
    pub clip_radius: f32,
}


/// Accumulates wall time between laps into the profiler's slots; the web has
/// no clock, so the laps are simply skipped there.
struct LapTimer {
    #[cfg(not(target_arch = "wasm32"))]
    last: Option<std::time::Instant>,
}

impl LapTimer {
    fn new(enabled: bool) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = enabled;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            last: enabled.then(std::time::Instant::now),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lap(&mut self, slot: &mut u64) {
        if let Some(t) = self.last {
            let now = std::time::Instant::now();
            *slot += (now - t).as_nanos() as u64;
            self.last = Some(now);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn lap(&mut self, _slot: &mut u64) {}
}

impl VVertex {
    fn new(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            x,
            y,
            u,
            v,
            stroke_dist: 0.0,
            clip_radius: 0.0,
        }
    }
    fn with_dist(x: f32, y: f32, u: f32, v: f32, stroke_dist: f32) -> Self {
        Self {
            x,
            y,
            u,
            v,
            stroke_dist,
            clip_radius: 0.0,
        }
    }
}

// Internal point with computed direction/miter info
#[derive(Clone, Copy, Default, Debug)]
struct VPoint {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,  // direction to next
    len: f32, // segment length
    dmx: f32,
    dmy: f32, // miter direction
    flags: u8,
}

const PT_CORNER: u8 = 1;
const PT_LEFT: u8 = 2;
const PT_BEVEL: u8 = 4;
const PT_INNERBEVEL: u8 = 8;

#[derive(Default, Debug)]
pub struct Tessellator {
    points: Vec<VPoint>,
    paths: Vec<SubPath>,
    cum_dists: Vec<f32>,
    /// Reusable sweep-line tessellator: keeps its event/edge/monotone
    /// buffers alive across fill() calls to avoid per-fill allocation churn.
    sweep: SweepTessellator,
    /// When set, `fill()` derives each contour's AA fill-side sign directly
    /// from its signed area instead of probing `point_in_fill_rule` around
    /// contour edges. Only enable this when contour winding is trustworthy
    /// (e.g. MVT map polygons whose ring orientation is spec-defined and
    /// normalized by the emitter): exterior and hole rings must be
    /// consistently, oppositely wound and non-self-intersecting.
    trust_fill_winding: bool,
    /// Bench-only escape hatch: disables the (behavior-identical) bbox
    /// prefilter in the fill-side sign probe so the original cost can be
    /// measured. Never set in production.
    disable_probe_bbox: bool,
    /// The fill rule the source actually asked for, when the caller knows it.
    /// `None` keeps the legacy guess in `fill()` for callers that build paths
    /// through `VectorPath` directly and never had a rule to state.
    fill_rule: Option<FillRule>,
}

/// Per-stage timing of a single `fill()` call, in nanoseconds.
/// Filled in by `fill_impl` when profiling is requested (bench/test only).
#[derive(Default, Clone, Copy, Debug)]
struct FillStats {
    joins_ns: u64,
    sign_ns: u64,
    body_emit_ns: u64,
    sweep_ns: u64,
    fringe_ns: u64,
}

#[derive(Debug)]
struct SubPath {
    first: usize,
    count: usize,
    closed: bool,
    winding: Winding,
    has_explicit_winding: bool,
    convex: bool,
    nbevel: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum FillRule {
    #[default]
    EvenOdd,
    NonZero,
}

impl Tessellator {
    /// State the fill rule the source asked for, which `fill()` then obeys
    /// instead of guessing.
    ///
    /// `None` restores the guess. The guess exists because this tessellator is
    /// also driven by callers that assemble a `VectorPath` by hand (charts, map
    /// tiles) and have no notion of an SVG `fill-rule`; it must stay, but it is
    /// wrong for SVG, whose `fill-rule` *defaults to nonzero* while the guess
    /// defaults to even-odd. That mismatch is half of why a pentagram used to
    /// fill wrong -- see `tests/self_intersecting_fill.rs`.
    ///
    /// Takes the *document's* rule ([`crate::path::FillRule`]); the enum used
    /// inside this module is a separate, `Default`-carrying copy that the glob
    /// import shadows, so the two are mapped explicitly here.
    pub fn set_fill_rule(&mut self, fill_rule: Option<crate::path::FillRule>) {
        self.fill_rule = fill_rule.map(|rule| match rule {
            crate::path::FillRule::NonZero => FillRule::NonZero,
            crate::path::FillRule::EvenOdd => FillRule::EvenOdd,
        });
    }

    /// Bounding box of flattened points: (min_x, min_y, max_x, max_y).
    /// Call after `flatten()`.
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
        let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
        for pt in &self.points {
            min_x = min_x.min(pt.x);
            min_y = min_y.min(pt.y);
            max_x = max_x.max(pt.x);
            max_y = max_y.max(pt.y);
        }
        (min_x, min_y, max_x, max_y)
    }

    pub fn flatten(&mut self, path: &VectorPath, tess_tol: f32) {
        self.points.clear();
        self.paths.clear();
        let dist_tol = 0.01;
        for cmd in &path.cmds {
            match *cmd {
                PathCmd::MoveTo(x, y) => {
                    self.paths.push(SubPath {
                        first: self.points.len(),
                        count: 0,
                        closed: false,
                        winding: Winding::CCW,
                        has_explicit_winding: false,
                        convex: false,
                        nbevel: 0,
                    });
                    self.add_point(x, y, PT_CORNER, dist_tol);
                }
                PathCmd::LineTo(x, y) => {
                    self.add_point(x, y, PT_CORNER, dist_tol);
                }
                PathCmd::BezierTo(cx1, cy1, cx2, cy2, x, y) => {
                    if let Some(last) = self.points.last().copied() {
                        self.tesselate_bezier(
                            last.x, last.y, cx1, cy1, cx2, cy2, x, y, 0, PT_CORNER, tess_tol,
                        );
                    }
                }
                PathCmd::Close => {
                    if let Some(p) = self.paths.last_mut() {
                        p.closed = true;
                    }
                }
                PathCmd::Winding(w) => {
                    if let Some(p) = self.paths.last_mut() {
                        p.winding = w;
                        p.has_explicit_winding = true;
                    }
                }
            }
        }
        self.prepare_points(dist_tol);
    }

    fn add_point(&mut self, x: f32, y: f32, flags: u8, dist_tol: f32) {
        if let Some(p) = self.paths.last_mut() {
            if p.count > 0 {
                if let Some(last) = self.points.last() {
                    let dx = x - last.x;
                    let dy = y - last.y;
                    if dx * dx + dy * dy < dist_tol * dist_tol {
                        self.points.last_mut().unwrap().flags |= flags;
                        return;
                    }
                }
            }
            self.points.push(VPoint {
                x,
                y,
                flags,
                ..Default::default()
            });
            p.count += 1;
        }
    }

    fn tesselate_bezier(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        x3: f32,
        y3: f32,
        x4: f32,
        y4: f32,
        level: usize,
        flags: u8,
        tess_tol: f32,
    ) {
        if level > 10 {
            return;
        }
        let x12 = (x1 + x2) * 0.5;
        let y12 = (y1 + y2) * 0.5;
        let x23 = (x2 + x3) * 0.5;
        let y23 = (y2 + y3) * 0.5;
        let x34 = (x3 + x4) * 0.5;
        let y34 = (y3 + y4) * 0.5;
        let x123 = (x12 + x23) * 0.5;
        let y123 = (y12 + y23) * 0.5;
        let dx = x4 - x1;
        let dy = y4 - y1;
        let d2 = ((x2 - x4) * dy - (y2 - y4) * dx).abs();
        let d3 = ((x3 - x4) * dy - (y3 - y4) * dx).abs();
        if (d2 + d3) * (d2 + d3) < tess_tol * (dx * dx + dy * dy) {
            self.add_point(x4, y4, flags, 0.01);
            return;
        }
        let x234 = (x23 + x34) * 0.5;
        let y234 = (y23 + y34) * 0.5;
        let x1234 = (x123 + x234) * 0.5;
        let y1234 = (y123 + y234) * 0.5;
        self.tesselate_bezier(
            x1,
            y1,
            x12,
            y12,
            x123,
            y123,
            x1234,
            y1234,
            level + 1,
            0,
            tess_tol,
        );
        self.tesselate_bezier(
            x1234,
            y1234,
            x234,
            y234,
            x34,
            y34,
            x4,
            y4,
            level + 1,
            flags,
            tess_tol,
        );
    }

    fn prepare_points(&mut self, dist_tol: f32) {
        for i in 0..self.paths.len() {
            let p = &mut self.paths[i];
            if p.count < 2 {
                continue;
            }
            let first = p.first;
            let count = p.count;
            // close duplicate check
            {
                let last = &self.points[first + count - 1];
                let fst = &self.points[first];
                let dx = fst.x - last.x;
                let dy = fst.y - last.y;
                if dx * dx + dy * dy < dist_tol * dist_tol {
                    p.count -= 1;
                    p.closed = true;
                }
            }
            let count = p.count;
            if count < 2 {
                continue;
            }
            // ensure correct winding
            if count > 2 {
                let area = poly_area(&self.points[first..first + count]);
                if p.has_explicit_winding {
                    match p.winding {
                        Winding::CCW => {
                            if area < 0.0 {
                                self.points[first..first + count].reverse();
                            }
                        }
                        Winding::CW => {
                            if area > 0.0 {
                                self.points[first..first + count].reverse();
                            }
                        }
                    }
                }
            }
            // compute segment directions + lengths
            for j in 0..count {
                let j1 = if j + 1 < count { j + 1 } else { 0 };
                let p0 = self.points[first + j];
                let p1 = self.points[first + j1];
                let mut dx = p1.x - p0.x;
                let mut dy = p1.y - p0.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-6 {
                    let il = 1.0 / len;
                    dx *= il;
                    dy *= il;
                }
                self.points[first + j].dx = dx;
                self.points[first + j].dy = dy;
                self.points[first + j].len = len;
            }
        }
    }

    fn calculate_joins(&mut self, w: f32, line_join: LineJoin, miter_limit: f32) {
        let iw = if w > 0.0 { 1.0 / w } else { 0.0 };
        for i in 0..self.paths.len() {
            let sp = &self.paths[i];
            let first = sp.first;
            let count = sp.count;
            if count < 2 {
                continue;
            }
            let mut nleft = 0usize;
            let mut nbevel = 0usize;
            for j in 0..count {
                let j0 = if j == 0 { count - 1 } else { j - 1 };
                let p0_dx = self.points[first + j0].dx;
                let p0_dy = self.points[first + j0].dy;
                let p1_dx = self.points[first + j].dx;
                let p1_dy = self.points[first + j].dy;
                let dlx0 = p0_dy;
                let dly0 = -p0_dx;
                let dlx1 = p1_dy;
                let dly1 = -p1_dx;
                let mut dmx = (dlx0 + dlx1) * 0.5;
                let mut dmy = (dly0 + dly1) * 0.5;
                let dmr2 = dmx * dmx + dmy * dmy;
                if dmr2 > 1e-6 {
                    let s = (1.0 / dmr2).min(600.0);
                    dmx *= s;
                    dmy *= s;
                }
                self.points[first + j].dmx = dmx;
                self.points[first + j].dmy = dmy;
                let mut flags = self.points[first + j].flags & PT_CORNER;
                let cross = p1_dx * p0_dy - p0_dx * p1_dy;
                if cross > 0.0 {
                    nleft += 1;
                    flags |= PT_LEFT;
                }
                let p0_len = self.points[first + j0].len;
                let p1_len = self.points[first + j].len;
                let limit = (p0_len.min(p1_len) * iw).max(1.01);
                if dmr2 * limit * limit < 1.0 {
                    flags |= PT_INNERBEVEL;
                }
                if (flags & PT_CORNER) != 0 {
                    if dmr2 * miter_limit * miter_limit < 1.0
                        || matches!(line_join, LineJoin::Bevel | LineJoin::Round)
                    {
                        flags |= PT_BEVEL;
                    }
                }
                if (flags & (PT_BEVEL | PT_INNERBEVEL)) != 0 {
                    nbevel += 1;
                }
                self.points[first + j].flags = flags;
            }
            self.paths[i].convex = nleft == count;
            self.paths[i].nbevel = nbevel;
        }
    }

    /// Generate stroke geometry into the provided vecs (clears them first).
    pub fn stroke(
        &mut self,
        w: f32,
        line_cap: LineCap,
        line_join: LineJoin,
        miter_limit: f32,
        aa: f32,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
    ) {
        self.stroke_ends(
            w,
            line_cap,
            line_cap,
            line_join,
            miter_limit,
            aa,
            verts,
            indices,
        );
    }

    /// Like `stroke` but with independent start/end caps, e.g. a butt cap at a
    /// clipped cut and a round cap at the true end of the same polyline.
    #[allow(clippy::too_many_arguments)]
    pub fn stroke_ends(
        &mut self,
        w: f32,
        start_cap: LineCap,
        end_cap: LineCap,
        line_join: LineJoin,
        miter_limit: f32,
        aa: f32,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
    ) {
        self.stroke_ends_impl(
            w, start_cap, end_cap, line_join, miter_limit, aa, verts, indices, None,
        );
    }

    /// `stroke_ends` that also records, per emitted vertex, the centerline
    /// anchor point it was expanded from. Consumers can rebuild the offset
    /// (`vertex - anchor`) and re-expand strokes in a vertex shader at a
    /// different width than the one baked here.
    #[allow(clippy::too_many_arguments)]
    pub fn stroke_ends_anchored(
        &mut self,
        w: f32,
        start_cap: LineCap,
        end_cap: LineCap,
        line_join: LineJoin,
        miter_limit: f32,
        aa: f32,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
        anchors: &mut Vec<[f32; 2]>,
    ) {
        anchors.clear();
        self.stroke_ends_impl(
            w,
            start_cap,
            end_cap,
            line_join,
            miter_limit,
            aa,
            verts,
            indices,
            Some(anchors),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn stroke_ends_impl(
        &mut self,
        w: f32,
        start_cap: LineCap,
        end_cap: LineCap,
        line_join: LineJoin,
        miter_limit: f32,
        aa: f32,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
        mut anchors: Option<&mut Vec<[f32; 2]>>,
    ) {
        let hw = w * 0.5 + aa * 0.5;
        self.calculate_joins(hw, line_join, miter_limit);
        verts.clear();
        indices.clear();
        let has_round_cap = self.paths.iter().any(|path| !path.closed)
            && (matches!(start_cap, LineCap::Round) || matches!(end_cap, LineCap::Round));
        let (u0, u1) = if aa > 0.0 || has_round_cap {
            (0.0, 1.0)
        } else {
            (0.5, 0.5)
        };
        for pi in 0..self.paths.len() {
            let sp = &self.paths[pi];
            let first = sp.first;
            let count = sp.count;
            if count < 2 {
                continue;
            }
            let is_loop = sp.closed;
            let base = verts.len() as u32;
            // compute cumulative distances for stroke_dist
            self.cum_dists.clear();
            self.cum_dists.resize(count, 0.0);
            {
                let mut d = 0.0f32;
                for j in 1..count {
                    d += self.points[first + j - 1].len;
                    self.cum_dists[j] = d;
                }
            }
            // caps for open paths
            if !is_loop {
                // Find a valid (non-degenerate) direction at the start of the path
                // by walking forward until we find two points far enough apart.
                let (ndx, ndy) = {
                    let mut dir = (0.0f32, 0.0f32);
                    for j in 1..count {
                        let dx = self.points[first + j].x - self.points[first].x;
                        let dy = self.points[first + j].y - self.points[first].y;
                        let len = (dx * dx + dy * dy).sqrt();
                        if len > 1e-6 {
                            dir = (dx / len, dy / len);
                            break;
                        }
                    }
                    dir
                };
                let p0 = self.points[first];
                self.emit_cap_start(
                    verts, indices, p0.x, p0.y, ndx, ndy, hw, u0, u1, start_cap,
                );
                // stamp stroke_dist=0 on cap verts
                let cap_end = verts.len();
                for v in &mut verts[base as usize..cap_end] {
                    v.stroke_dist = 0.0;
                }
                if let Some(anchors) = anchors.as_deref_mut() {
                    anchors.resize(verts.len(), [p0.x, p0.y]);
                }
            }
            // body
            let (s, e) = if is_loop { (0, count) } else { (1, count - 1) };
            for j in s..e {
                let j0 = if j == 0 { count - 1 } else { j - 1 };
                let p0 = self.points[first + j0];
                let p1 = self.points[first + j];
                let dist = self.cum_dists[j];
                let flags = p1.flags;
                if (flags & (PT_BEVEL | PT_INNERBEVEL)) != 0 {
                    let vi_before = verts.len();
                    self.emit_bevel_join(verts, indices, p0, p1, hw, hw, u0, u1);
                    for v in &mut verts[vi_before..] {
                        v.stroke_dist = dist;
                    }
                    if let Some(anchors) = anchors.as_deref_mut() {
                        anchors.resize(verts.len(), [p1.x, p1.y]);
                    }
                } else {
                    let vi = verts.len() as u32;
                    verts.push(VVertex::with_dist(
                        p1.x + p1.dmx * hw,
                        p1.y + p1.dmy * hw,
                        u0,
                        1.0,
                        dist,
                    ));
                    verts.push(VVertex::with_dist(
                        p1.x - p1.dmx * hw,
                        p1.y - p1.dmy * hw,
                        u1,
                        1.0,
                        dist,
                    ));
                    if let Some(anchors) = anchors.as_deref_mut() {
                        anchors.resize(verts.len(), [p1.x, p1.y]);
                    }
                    if vi >= base + 2 {
                        indices.push(vi - 2);
                        indices.push(vi - 1);
                        indices.push(vi);
                        indices.push(vi - 1);
                        indices.push(vi + 1);
                        indices.push(vi);
                    }
                }
            }
            if !is_loop {
                // end cap: find a valid direction by walking backward from the end
                let p1 = self.points[first + count - 1];
                let (ndx, ndy) = {
                    let mut dir = (0.0f32, 0.0f32);
                    for j in (0..count - 1).rev() {
                        let dx = p1.x - self.points[first + j].x;
                        let dy = p1.y - self.points[first + j].y;
                        let len = (dx * dx + dy * dy).sqrt();
                        if len > 1e-6 {
                            dir = (dx / len, dy / len);
                            break;
                        }
                    }
                    dir
                };
                let vi_before = verts.len();
                self.emit_cap_end(
                    verts, indices, p1.x, p1.y, ndx, ndy, hw, aa, u0, u1, end_cap,
                );
                let total_dist = self.cum_dists[count - 1];
                for v in &mut verts[vi_before..] {
                    v.stroke_dist = total_dist;
                }
                if let Some(anchors) = anchors.as_deref_mut() {
                    anchors.resize(verts.len(), [p1.x, p1.y]);
                }
            } else {
                // close loop: connect last pair to first pair
                let vi = verts.len() as u32;
                if vi >= base + 4 {
                    indices.push(vi - 2);
                    indices.push(vi - 1);
                    indices.push(base);
                    indices.push(vi - 1);
                    indices.push(base + 1);
                    indices.push(base);
                }
            }
        }
    }

    fn emit_cap_start(
        &self,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
        px: f32,
        py: f32,
        dx: f32,
        dy: f32,
        w: f32,
        u0: f32,
        u1: f32,
        cap: LineCap,
    ) {
        let dlx = dy;
        let dly = -dx;
        match cap {
            LineCap::Butt => {
                verts.push(VVertex::new(px + dlx * w, py + dly * w, u0, 1.0));
                verts.push(VVertex::new(px - dlx * w, py - dly * w, u1, 1.0));
            }
            LineCap::Square => {
                verts.push(VVertex::new(px + dlx * w, py + dly * w, u0, 1.0));
                verts.push(VVertex::new(px - dlx * w, py - dly * w, u1, 1.0));
            }
            LineCap::Round => {
                let vi = verts.len() as u32;
                verts.push(VVertex::new(
                    px + dlx * w - dx * w,
                    py + dly * w - dy * w,
                    u0,
                    2.0,
                ));
                verts.push(VVertex::new(
                    px - dlx * w - dx * w,
                    py - dly * w - dy * w,
                    u1,
                    2.0,
                ));
                verts.push(VVertex::new(px + dlx * w, py + dly * w, u0, 1.0));
                verts.push(VVertex::new(px - dlx * w, py - dly * w, u1, 1.0));
                indices.extend_from_slice(&[
                    vi,
                    vi + 1,
                    vi + 2,
                    vi + 1,
                    vi + 3,
                    vi + 2,
                ]);
            }
        }
    }

    fn emit_cap_end(
        &self,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
        px: f32,
        py: f32,
        dx: f32,
        dy: f32,
        w: f32,
        aa: f32,
        u0: f32,
        u1: f32,
        cap: LineCap,
    ) {
        let dlx = dy;
        let dly = -dx;
        let vi = verts.len() as u32;
        match cap {
            LineCap::Butt => {
                verts.push(VVertex::new(px + dlx * w, py + dly * w, u0, 1.0));
                verts.push(VVertex::new(px - dlx * w, py - dly * w, u1, 1.0));
                verts.push(VVertex::new(
                    px + dlx * w + dx * aa,
                    py + dly * w + dy * aa,
                    u0,
                    0.0,
                ));
                verts.push(VVertex::new(
                    px - dlx * w + dx * aa,
                    py - dly * w + dy * aa,
                    u1,
                    0.0,
                ));
            }
            LineCap::Square => {
                verts.push(VVertex::new(px + dlx * w, py + dly * w, u0, 1.0));
                verts.push(VVertex::new(px - dlx * w, py - dly * w, u1, 1.0));
                verts.push(VVertex::new(
                    px + dlx * w + dx * (w - aa),
                    py + dly * w + dy * (w - aa),
                    u0,
                    0.0,
                ));
                verts.push(VVertex::new(
                    px - dlx * w + dx * (w - aa),
                    py - dly * w + dy * (w - aa),
                    u1,
                    0.0,
                ));
            }
            LineCap::Round => {
                verts.push(VVertex::new(px + dlx * w, py + dly * w, u0, 1.0));
                verts.push(VVertex::new(px - dlx * w, py - dly * w, u1, 1.0));
                if vi >= 2 {
                    indices.push(vi - 2);
                    indices.push(vi - 1);
                    indices.push(vi);
                    indices.push(vi - 1);
                    indices.push(vi + 1);
                    indices.push(vi);
                }
                verts.push(VVertex::new(
                    px + dlx * w + dx * w,
                    py + dly * w + dy * w,
                    u0,
                    2.0,
                ));
                verts.push(VVertex::new(
                    px - dlx * w + dx * w,
                    py - dly * w + dy * w,
                    u1,
                    2.0,
                ));
                indices.extend_from_slice(&[
                    vi,
                    vi + 1,
                    vi + 2,
                    vi + 1,
                    vi + 3,
                    vi + 2,
                ]);
                return;
            }
        }
        // connect cap end to previous pair (Butt/Square only)
        if vi >= 2 {
            indices.push(vi - 2);
            indices.push(vi - 1);
            indices.push(vi);
            indices.push(vi - 1);
            indices.push(vi + 1);
            indices.push(vi);
        }
        // stitch cap triangles (Butt/Square: simple quad strip)
        let n = (verts.len() as u32 - vi) / 2;
        for i in 1..n {
            let a = vi + (i - 1) * 2;
            let b = vi + i * 2;
            indices.push(a);
            indices.push(a + 1);
            indices.push(b);
            indices.push(a + 1);
            indices.push(b + 1);
            indices.push(b);
        }
    }

    fn emit_bevel_join(
        &self,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
        p0: VPoint,
        p1: VPoint,
        lw: f32,
        rw: f32,
        u0: f32,
        u1: f32,
    ) {
        let vi = verts.len() as u32;
        let dlx0 = p0.dy;
        let dly0 = -p0.dx;
        let dlx1 = p1.dy;
        let dly1 = -p1.dx;
        if (p1.flags & PT_LEFT) != 0 {
            let lx0 = p1.x + dlx0 * lw;
            let ly0 = p1.y + dly0 * lw;
            let lx1 = p1.x + dlx1 * lw;
            let ly1 = p1.y + dly1 * lw;
            verts.push(VVertex::new(lx0, ly0, u0, 1.0));
            verts.push(VVertex::new(p1.x - dlx0 * rw, p1.y - dly0 * rw, u1, 1.0));
            verts.push(VVertex::new(lx1, ly1, u0, 1.0));
            verts.push(VVertex::new(p1.x - dlx1 * rw, p1.y - dly1 * rw, u1, 1.0));
        } else {
            let rx0 = p1.x - dlx0 * rw;
            let ry0 = p1.y - dly0 * rw;
            let rx1 = p1.x - dlx1 * rw;
            let ry1 = p1.y - dly1 * rw;
            verts.push(VVertex::new(p1.x + dlx0 * lw, p1.y + dly0 * lw, u0, 1.0));
            verts.push(VVertex::new(rx0, ry0, u1, 1.0));
            verts.push(VVertex::new(p1.x + dlx1 * lw, p1.y + dly1 * lw, u0, 1.0));
            verts.push(VVertex::new(rx1, ry1, u1, 1.0));
        }
        // connect to previous pair and within bevel
        if vi >= 2 {
            indices.push(vi - 2);
            indices.push(vi - 1);
            indices.push(vi);
            indices.push(vi - 1);
            indices.push(vi + 1);
            indices.push(vi);
        }
        indices.push(vi);
        indices.push(vi + 1);
        indices.push(vi + 2);
        indices.push(vi + 1);
        indices.push(vi + 3);
        indices.push(vi + 2);
    }

    /// Opt-in fast path for `fill()`: derive AA fill-side signs from contour
    /// winding (signed area) instead of point-in-fill probing. See the field
    /// doc on `trust_fill_winding` for when this is safe.
    pub fn set_trust_fill_winding(&mut self, on: bool) {
        self.trust_fill_winding = on;
    }

    /// Generate fill geometry into the provided vecs (clears them first).
    pub fn fill(
        &mut self,
        aa: f32,
        line_join: LineJoin,
        miter_limit: f32,
        gpu_expand_fill: bool,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
    ) {
        self.fill_impl(aa, line_join, miter_limit, gpu_expand_fill, verts, indices, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_impl(
        &mut self,
        aa: f32,
        line_join: LineJoin,
        miter_limit: f32,
        gpu_expand_fill: bool,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
        mut stats: Option<&mut FillStats>,
    ) {
        // Env-gated fixture capture: append the flattened input of this fill
        // call to MAKEPAD_TESS_DUMP for offline benchmark replay.
        #[cfg(not(target_arch = "wasm32"))]
        fixture_dump::dump_fill(self, aa, line_join, miter_limit, gpu_expand_fill);

        let mut lap_timer = LapTimer::new(stats.is_some());
        let mut lap = move |slot: &mut u64| {
            lap_timer.lap(slot);
        };

        let woff = aa * 0.5;
        self.calculate_joins(woff, line_join, miter_limit);
        verts.clear();
        indices.clear();
        if let Some(stats) = stats.as_deref_mut() {
            lap(&mut stats.joins_ns);
        }

        // Collect valid subpaths (>= 3 points) with per-contour AA side sign.
        // sign > 0: fill side is along +dmx/+dmy, sign < 0: fill side is along -dmx/-dmy.
        // We derive this from a non-zero fill test around each contour edge.
        let mut valid_paths: Vec<(usize, usize, f32)> = Vec::new();
        for pi in 0..self.paths.len() {
            let sp = &self.paths[pi];
            if sp.count >= 3 {
                valid_paths.push((sp.first, sp.count, 1.0));
            }
        }

        if valid_paths.is_empty() {
            return;
        }

        // Did the caller hand us contours that state their own direction? This
        // is the legacy stand-in for a fill rule, and it is still what the AA
        // inset below keys off.
        let has_stated_winding = self
            .paths
            .iter()
            .any(|sp| sp.count >= 3 && sp.has_explicit_winding);
        // A rule the caller actually knows beats the guess. SVG always knows
        // (`fill-rule` defaults to nonzero), so this is how a browser-correct
        // answer reaches the sweep; see `set_fill_rule`.
        let fill_rule = self.fill_rule.unwrap_or(if has_stated_winding {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        });
        let body_inset_woff = if has_stated_winding {
            woff
        } else {
            // Implicit font-like outlines are fragile under inward body shrink.
            // Keep body on-edge and let the fringe provide AA falloff.
            //
            // Deliberately keyed on the GUESS and not on `fill_rule`: this is an
            // antialiasing choice about how the contour was authored, not about
            // how regions are classified. Re-keying it on the now caller-stated
            // rule would quietly inset the body of every SVG fill in the
            // codebase, which is a visual change with no bug behind it.
            0.0
        };

        self.compute_fill_signs(&mut valid_paths, fill_rule);
        if let Some(stats) = stats.as_deref_mut() {
            lap(&mut stats.sign_ns);
        }

        // Emit fill body vertices.
        // In regular mode we inset by woff like NanoVG. In GPU-expand mode
        // we keep vertices on the edge and let the vertex shader place them.
        let fill_base = verts.len() as u32;
        for &(first, count, sign) in &valid_paths {
            for j in 0..count {
                let pt = self.points[first + j];
                if gpu_expand_fill {
                    let nx = -pt.dmx * sign;
                    let ny = -pt.dmy * sign;
                    verts.push(VVertex {
                        x: pt.x,
                        y: pt.y,
                        u: 0.5,
                        v: nx,
                        stroke_dist: ny,
                        clip_radius: 0.0,
                    });
                } else {
                    verts.push(VVertex::new(
                        pt.x + pt.dmx * body_inset_woff * sign,
                        pt.y + pt.dmy * body_inset_woff * sign,
                        0.5,
                        1.0,
                    ));
                }
            }
        }
        if let Some(stats) = stats.as_deref_mut() {
            lap(&mut stats.body_emit_ns);
        }

        // Feed edges from ALL subpaths into a single sweep-line tessellator,
        // which classifies regions with `fill_rule`.
        //
        // The sweep derives its events solely from endpoints it is handed, so
        // two edges crossing in their *interiors* generate no event at all and
        // every region downstream of the crossing is attributed wrongly. So
        // planarise first: `split_edges_at_crossings` cuts each crossing into a
        // shared endpoint the sweep already understands.
        //
        // When it finds no crossing -- the overwhelmingly common case, and every
        // map tile -- nothing is cut and the ring fast path runs unchanged, so
        // the allocation-free `push_ring` still serves those fills. Only a
        // genuinely self-intersecting path pays for the per-edge path.
        {
            let mut edges: Vec<FEdge> = Vec::new();
            let mut offset = 0usize;
            for &(_first, count, _) in &valid_paths {
                for i in 0..count {
                    let j = (i + 1) % count;
                    let vi = verts[fill_base as usize + offset + i];
                    let vj = verts[fill_base as usize + offset + j];
                    // Preserve original contour order for non-zero winding.
                    edges.push(FEdge {
                        a: FPoint::new(vi.x, vi.y),
                        a_index: fill_base + (offset + i) as u32,
                        b: FPoint::new(vj.x, vj.y),
                        b_index: fill_base + (offset + j) as u32,
                    });
                }
                offset += count;
            }
            // A crossing always interns at least one new vertex, so the vertex
            // count is exactly the signal for whether anything was cut.
            let verts_before = verts.len();
            split_edges_at_crossings(&mut edges, verts, gpu_expand_fill);
            let planarised = verts.len() != verts_before;

            let tess = &mut self.sweep;
            tess.reset(fill_rule);
            if planarised {
                for edge in &edges {
                    tess.push_edge(edge.a, edge.a_index, edge.b, edge.b_index);
                }
            } else {
                let all_fill_verts = &verts[fill_base as usize..];
                let mut offset = 0usize;
                for &(_first, count, _) in &valid_paths {
                    tess.push_ring(
                        &all_fill_verts[offset..offset + count],
                        fill_base + offset as u32,
                    );
                    offset += count;
                }
            }
            tess.tessellate_into(indices);
        }
        if let Some(stats) = stats.as_deref_mut() {
            lap(&mut stats.sweep_ns);
        }

        if woff > 0.0 {
            self.emit_fill_fringe(
                &valid_paths,
                woff,
                body_inset_woff,
                gpu_expand_fill,
                verts,
                indices,
            );
        }
        if let Some(stats) = stats.as_deref_mut() {
            lap(&mut stats.fringe_ns);
        }
    }

    /// AA fringe: inner vertex at body edge (opaque, u=0.5),
    /// outer vertex also at body edge but tagged with the outward
    /// normal in (v, stroke_dist) so the vertex shader can expand
    /// it to the correct screen-space width. Appends self-contained
    /// strips (indices only reference vertices pushed here).
    #[allow(clippy::too_many_arguments)]
    fn emit_fill_fringe(
        &self,
        valid_paths: &[(usize, usize, f32)],
        woff: f32,
        body_inset_woff: f32,
        gpu_expand_fill: bool,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
    ) {
        for &(first, count, sign) in valid_paths {
            let fringe_base = verts.len() as u32;
            for j in 0..count {
                let p1 = self.points[first + j];
                if gpu_expand_fill {
                    // Anchor both fringe vertices at the edge and encode
                    // outward normal for GPU-side fringe placement.
                    let bx = p1.x;
                    let by = p1.y;
                    let nx = -p1.dmx * sign;
                    let ny = -p1.dmy * sign;
                    verts.push(VVertex {
                        x: bx,
                        y: by,
                        u: 0.5,
                        v: nx,
                        stroke_dist: ny,
                        clip_radius: 0.0,
                    });
                    verts.push(VVertex {
                        x: bx,
                        y: by,
                        u: 0.0,
                        v: nx,
                        stroke_dist: ny,
                        clip_radius: 0.0,
                    });
                } else {
                    // Classic NanoVG-style physical fringe geometry.
                    verts.push(VVertex::new(
                        p1.x + p1.dmx * body_inset_woff * sign,
                        p1.y + p1.dmy * body_inset_woff * sign,
                        0.5,
                        1.0,
                    ));
                    verts.push(VVertex::new(
                        p1.x - p1.dmx * woff * sign,
                        p1.y - p1.dmy * woff * sign,
                        0.0,
                        1.0,
                    ));
                }
            }
            for j in 0..count as u32 {
                let j1 = if j + 1 < count as u32 { j + 1 } else { 0 };
                let a = fringe_base + j * 2;
                let b = fringe_base + j1 * 2;
                indices.push(a);
                indices.push(a + 1);
                indices.push(b);
                indices.push(a + 1);
                indices.push(b + 1);
                indices.push(b);
            }
        }
    }

    /// Append ONLY the AA fringe of the flattened path into `verts` /
    /// `indices` (no body triangulation): the fast path for pre-baked fill
    /// bodies, whose edge AA must stay identical to the runtime fill's.
    /// Same sign derivation, joins math and vertex semantics as `fill()`;
    /// unlike `fill()` this appends instead of clearing. Call `flatten`
    /// first.
    pub fn fill_fringe_into(
        &mut self,
        aa: f32,
        line_join: LineJoin,
        miter_limit: f32,
        gpu_expand_fill: bool,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
    ) {
        let woff = aa * 0.5;
        if woff <= 0.0 {
            return;
        }
        self.calculate_joins(woff, line_join, miter_limit);
        let mut valid_paths: Vec<(usize, usize, f32)> = Vec::new();
        for sp in &self.paths {
            if sp.count >= 3 {
                valid_paths.push((sp.first, sp.count, 1.0));
            }
        }
        if valid_paths.is_empty() {
            return;
        }
        // Same split as `fill_into`: the fill-side signs must agree with the
        // body that path emitted, so they follow the caller-stated rule, while
        // the AA inset stays keyed on the legacy guess.
        let has_stated_winding = self
            .paths
            .iter()
            .any(|sp| sp.count >= 3 && sp.has_explicit_winding);
        let fill_rule = self.fill_rule.unwrap_or(if has_stated_winding {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        });
        let body_inset_woff = if has_stated_winding { woff } else { 0.0 };
        self.compute_fill_signs(&mut valid_paths, fill_rule);
        self.emit_fill_fringe(
            &valid_paths,
            woff,
            body_inset_woff,
            gpu_expand_fill,
            verts,
            indices,
        );
    }

    /// Per-contour AA fill-side signs: trusted-winding fast path when
    /// enabled (multi-contour fills), otherwise the probing baseline.
    fn compute_fill_signs(&self, valid_paths: &mut [(usize, usize, f32)], fill_rule: FillRule) {
        // Fast path: with trustworthy winding (map polygons: exterior rings
        // consistently wound, holes opposite) the fill-side sign is the SAME
        // for every contour of the fill and equals the exterior ring's
        // orientation sign: flipping a hole's winding flips both its miter
        // normals AND which side is filled, so the sign cancels back to the
        // exterior's. The exterior is the largest-|area| contour (a hole is
        // always smaller than the ring that contains it). O(n) total,
        // replacing O(edges x total_verts) probing.
        //
        // Single-contour fills keep the probe: it is O(n) there anyway, and
        // for self-intersecting slivers (bowties) it picks the locally
        // filled side, which net orientation cannot.
        if self.trust_fill_winding && valid_paths.len() >= 2 {
            let mut areas: Vec<f32> = Vec::with_capacity(valid_paths.len());
            let mut max_area = -1.0f32;
            let mut fill_sign = 1.0f32;
            for &(first, count, _) in valid_paths.iter() {
                let area = poly_area(&self.points[first..first + count]);
                areas.push(area);
                if area.abs() > max_area {
                    max_area = area.abs();
                    fill_sign = if area >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            for (vp, &area) in valid_paths.iter_mut().zip(&areas) {
                vp.2 = fill_sign;
                // Near-degenerate hole-wound slivers are ambiguous: a true
                // hole takes the exterior's sign, but an orphan sliver
                // OUTSIDE the exterior (emission glitch) fills like an
                // island and needs its own orientation. Mark for probing.
                let sliver = area.abs() < max_area * 1e-2;
                let hole_wound = (area >= 0.0) != (fill_sign >= 0.0);
                if sliver && hole_wound {
                    vp.2 = 0.0; // sentinel: resolve by probe below
                }
            }
            if valid_paths.iter().any(|vp| vp.2 == 0.0) {
                self.probe_fill_signs_inner(valid_paths, fill_rule, true);
            }
        } else {
            self.probe_fill_signs(valid_paths, fill_rule);
        }
    }

    /// Baseline fill-side sign derivation: probe both sides of contour
    /// edges with the fill rule until one edge gives a decisive answer.
    /// Contours the probe cannot decide fall back to orientation.
    fn probe_fill_signs(&self, valid_paths: &mut [(usize, usize, f32)], fill_rule: FillRule) {
        self.probe_fill_signs_inner(valid_paths, fill_rule, false)
    }

    /// `only_sentinels`: probe only contours whose sign is the 0.0 sentinel
    /// (set by the trusted-winding fast path for ambiguous slivers).
    fn probe_fill_signs_inner(
        &self,
        valid_paths: &mut [(usize, usize, f32)],
        fill_rule: FillRule,
        only_sentinels: bool,
    ) {
        // Per-contour bounding boxes: a probe point strictly outside a closed
        // contour's bbox has zero winding contribution from it, so
        // point_in_fill_rule can skip that contour outright.
        let bboxes: Vec<[f32; 4]> = valid_paths
            .iter()
            .map(|&(first, count, _)| {
                let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
                let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
                for pt in &self.points[first..first + count] {
                    min_x = min_x.min(pt.x);
                    min_y = min_y.min(pt.y);
                    max_x = max_x.max(pt.x);
                    max_y = max_y.max(pt.y);
                }
                [min_x, min_y, max_x, max_y]
            })
            .collect();
        let bboxes_opt = if self.disable_probe_bbox {
            None
        } else {
            Some(bboxes.as_slice())
        };
        for i in 0..valid_paths.len() {
            if only_sentinels && valid_paths[i].2 != 0.0 {
                continue;
            }
            let (first, count, _) = valid_paths[i];
            let mut sign = 1.0f32;
            let mut found = false;

            for j in 0..count {
                let j1 = (j + 1) % count;
                let p0 = self.points[first + j];
                let p1 = self.points[first + j1];
                let mut ex = p1.x - p0.x;
                let mut ey = p1.y - p0.y;
                let e2 = ex * ex + ey * ey;
                if e2 <= 1e-12 {
                    continue;
                }

                let inv_e = 1.0 / e2.sqrt();
                ex *= inv_e;
                ey *= inv_e;
                let nx = ey;
                let ny = -ex;
                let mx = (p0.x + p1.x) * 0.5;
                let my = (p0.y + p1.y) * 0.5;
                let local_len = p0.len.max(p1.len);
                let base_eps = (local_len * 1e-3).max(1e-4);
                let eps_scales = [1.0f32, 4.0, 16.0];

                for s in eps_scales {
                    let eps = base_eps * s;
                    let plus_filled = point_in_fill_rule(
                        mx + nx * eps,
                        my + ny * eps,
                        &self.points,
                        valid_paths,
                        bboxes_opt,
                        fill_rule,
                    );
                    let minus_filled = point_in_fill_rule(
                        mx - nx * eps,
                        my - ny * eps,
                        &self.points,
                        valid_paths,
                        bboxes_opt,
                        fill_rule,
                    );

                    if plus_filled != minus_filled {
                        sign = if plus_filled { 1.0 } else { -1.0 };
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            if !found {
                // Fallback to contour orientation if local fill-side probe was inconclusive.
                let area = poly_area(&self.points[first..first + count]);
                sign = if area >= 0.0 { 1.0 } else { -1.0 };
            }
            valid_paths[i].2 = sign;
        }
    }

    /// Generate shadow geometry for arbitrary filled shapes.
    /// Like fill() but with a wide fringe (3*blur) for gaussian falloff.
    /// The v coordinate encodes normalized distance from edge: 1.0 = inside, 0.0 = outer limit.
    /// stroke_dist is repurposed to carry the blur radius for the shader.
    pub fn fill_shadow(
        &mut self,
        blur: f32,
        line_join: LineJoin,
        miter_limit: f32,
        verts: &mut Vec<VVertex>,
        indices: &mut Vec<u32>,
    ) {
        let expand = blur * 3.0;
        // Ensure CCW winding so that dmx/dmy normals point outward.
        // prepare_points no longer auto-corrects winding for paths
        // without explicit winding, so we must enforce it here.
        for pi in 0..self.paths.len() {
            let sp = &self.paths[pi];
            let first = sp.first;
            let count = sp.count;
            if count >= 3 {
                let area = poly_area(&self.points[first..first + count]);
                if area < 0.0 {
                    self.points[first..first + count].reverse();
                }
            }
        }
        self.calculate_joins(expand, line_join, miter_limit);
        verts.clear();
        indices.clear();
        for pi in 0..self.paths.len() {
            let sp = &self.paths[pi];
            let first = sp.first;
            let count = sp.count;
            if count < 3 {
                continue;
            }
            // Determine outward direction sign from contour winding.
            // For CCW (area >= 0), dmx/dmy points outward.
            // For CW (area < 0), dmx/dmy points inward.
            // We want to push fringe outward, so use sign * dmx.
            let area = poly_area(&self.points[first..first + count]);
            let sign: f32 = if area >= 0.0 { 1.0 } else { -1.0 };
            // convex fan at original path positions (fully opaque, v=1.0)
            let base = verts.len() as u32;
            for j in 0..count {
                let pt = self.points[first + j];
                // u=0.5 (opaque in AA formula), v=1.0 (inside)
                let mut v = VVertex::new(pt.x, pt.y, 0.5, 1.0);
                v.stroke_dist = blur; // carry blur radius
                verts.push(v);
            }
            for j in 2..count as u32 {
                indices.push(base);
                indices.push(base + j - 1);
                indices.push(base + j);
            }
            // wide fringe for shadow falloff
            let fringe_base = verts.len() as u32;
            for j in 0..count {
                let p1 = self.points[first + j];
                // inner: at path edge (v=1.0, will be fully opaque)
                let mut vi = VVertex::new(p1.x, p1.y, 0.5, 1.0);
                vi.stroke_dist = blur;
                verts.push(vi);
                // outer: pushed outward, clamped to expand distance
                let mut ox = sign * p1.dmx * expand;
                let mut oy = sign * p1.dmy * expand;
                let ol = (ox * ox + oy * oy).sqrt();
                if ol > expand {
                    let s = expand / ol;
                    ox *= s;
                    oy *= s;
                }
                let mut vo = VVertex::new(p1.x + ox, p1.y + oy, 0.0, 0.0);
                vo.stroke_dist = blur;
                verts.push(vo);
            }
            // stitch fringe strip
            for j in 0..count as u32 {
                let j1 = if j + 1 < count as u32 { j + 1 } else { 0 };
                let a = fringe_base + j * 2;
                let b = fringe_base + j1 * 2;
                indices.push(a);
                indices.push(a + 1);
                indices.push(b);
                indices.push(a + 1);
                indices.push(b + 1);
                indices.push(b);
            }
        }
    }
}

/// Compute per-vertex clip_radius from the triangle index buffer.
/// For each vertex, this is the maximum distance to any other vertex
/// it shares a triangle with. This allows the vertex shader to skip
/// triangles that are entirely outside the clip rect.
pub fn compute_clip_radii(verts: &mut [VVertex], indices: &[u32]) {
    // Process triangles: for each triangle (a, b, c), update each vertex's
    // clip_radius to be the max distance to any other vertex in that triangle.
    let mut i = 0;
    while i + 2 < indices.len() {
        let ia = indices[i] as usize;
        let ib = indices[i + 1] as usize;
        let ic = indices[i + 2] as usize;
        if ia < verts.len() && ib < verts.len() && ic < verts.len() {
            let ax = verts[ia].x;
            let ay = verts[ia].y;
            let bx = verts[ib].x;
            let by = verts[ib].y;
            let cx = verts[ic].x;
            let cy = verts[ic].y;

            let dab = ((ax - bx) * (ax - bx) + (ay - by) * (ay - by)).sqrt();
            let dac = ((ax - cx) * (ax - cx) + (ay - cy) * (ay - cy)).sqrt();
            let dbc = ((bx - cx) * (bx - cx) + (by - cy) * (by - cy)).sqrt();

            let ra = dab.max(dac);
            let rb = dab.max(dbc);
            let rc = dac.max(dbc);

            if ra > verts[ia].clip_radius {
                verts[ia].clip_radius = ra;
            }
            if rb > verts[ib].clip_radius {
                verts[ib].clip_radius = rb;
            }
            if rc > verts[ic].clip_radius {
                verts[ic].clip_radius = rc;
            }
        }
        i += 3;
    }
}

fn poly_area(pts: &[VPoint]) -> f32 {
    let mut area = 0.0;
    let n = pts.len();
    for i in 2..n {
        area += (pts[i].x - pts[0].x) * (pts[i - 1].y - pts[0].y)
            - (pts[i - 1].x - pts[0].x) * (pts[i].y - pts[0].y);
    }
    area * 0.5
}

fn contour_winding_at_point(px: f32, py: f32, pts: &[VPoint], first: usize, count: usize) -> i32 {
    if count < 3 {
        return 0;
    }
    let mut winding = 0i32;
    for i in 0..count {
        let j = (i + 1) % count;
        let x0 = pts[first + i].x;
        let y0 = pts[first + i].y;
        let x1 = pts[first + j].x;
        let y1 = pts[first + j].y;

        if y0 <= py {
            if y1 > py {
                let is_left = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0);
                if is_left > 0.0 {
                    winding += 1;
                }
            }
        } else if y1 <= py {
            let is_left = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0);
            if is_left < 0.0 {
                winding -= 1;
            }
        }
    }
    winding
}

fn point_in_fill_rule(
    px: f32,
    py: f32,
    pts: &[VPoint],
    contours: &[(usize, usize, f32)],
    bboxes: Option<&[[f32; 4]]>,
    fill_rule: FillRule,
) -> bool {
    let mut winding = 0i32;
    for (k, &(first, count, _)) in contours.iter().enumerate() {
        if let Some(bboxes) = bboxes {
            // A point strictly outside a closed contour's bbox winds zero
            // around it: skipping is behavior-identical, just faster.
            let bb = &bboxes[k];
            if px < bb[0] || py < bb[1] || px > bb[2] || py > bb[3] {
                continue;
            }
        }
        winding += contour_winding_at_point(px, py, pts, first, count);
    }
    match fill_rule {
        FillRule::NonZero => winding != 0,
        FillRule::EvenOdd => (winding & 1) != 0,
    }
}

/// Env-gated fixture capture for offline tessellation benchmarks.
/// `MAKEPAD_TESS_DUMP=/path/file.tessfix` appends one length-prefixed binary
/// record per `fill()` call: the fill parameters plus every flattened contour
/// (post `flatten`/`prepare_points`, pre `calculate_joins`).
#[cfg(not(target_arch = "wasm32"))]
mod fixture_dump {
    use super::Tessellator;
    use crate::path::{LineJoin, Winding};
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    pub(super) const MAGIC: u32 = 0x5445_5346; // "TESF"
    pub(super) const VERSION: u32 = 1;

    static DUMP: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

    fn writer() -> Option<&'static Mutex<std::fs::File>> {
        DUMP.get_or_init(|| {
            let path = std::env::var("MAKEPAD_TESS_DUMP").ok()?;
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| {
                    eprintln!("MAKEPAD_TESS_DUMP: cannot open {path}: {e}");
                    e
                })
                .ok()?;
            Some(Mutex::new(file))
        })
        .as_ref()
    }

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(super) fn dump_fill(
        tess: &Tessellator,
        aa: f32,
        line_join: LineJoin,
        miter_limit: f32,
        gpu_expand_fill: bool,
    ) {
        let Some(file) = writer() else {
            return;
        };
        let mut buf = Vec::with_capacity(64 + tess.points.len() * 12);
        push_u32(&mut buf, MAGIC);
        push_u32(&mut buf, VERSION);
        push_f32(&mut buf, aa);
        push_u32(
            &mut buf,
            match line_join {
                LineJoin::Miter => 0,
                LineJoin::Round => 1,
                LineJoin::Bevel => 2,
            },
        );
        push_f32(&mut buf, miter_limit);
        push_u32(&mut buf, gpu_expand_fill as u32);
        push_u32(&mut buf, tess.paths.len() as u32);
        for sp in &tess.paths {
            push_u32(&mut buf, sp.count as u32);
            let flags = (sp.closed as u32)
                | ((sp.has_explicit_winding as u32) << 1)
                | ((matches!(sp.winding, Winding::CW) as u32) << 2);
            push_u32(&mut buf, flags);
            for pt in &tess.points[sp.first..sp.first + sp.count] {
                push_f32(&mut buf, pt.x);
                push_f32(&mut buf, pt.y);
                push_u32(&mut buf, pt.flags as u32);
            }
        }
        if let Ok(mut file) = file.lock() {
            let _ = file.write_all(&buf);
        }
    }
}

// ---- Sweep-line monotone polygon tessellator (ported from bender) ----
// Decomposes concave polygons into monotone sub-polygons with a sweep line,
// then triangulates each.
//
// It copes with SELF-INTERSECTING input only because `split_edges_at_crossings`
// below hands it an already-planar edge set. On its own it does not: every
// event it processes comes from an endpoint it was given, so two edges crossing
// in their interiors are invisible to it. This comment used to claim the
// opposite, which is what sent the pentagram investigation off to the GPU
// shader instead of here.

// Minimal 2D point for the tessellator
#[derive(Clone, Copy, Debug)]
struct FPoint {
    x: f32,
    y: f32,
}

impl FPoint {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl PartialEq for FPoint {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl PartialOrd for FPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Match bender geometry Point ordering: x first, then y.
        match self.x.partial_cmp(&other.x) {
            Some(std::cmp::Ordering::Equal) => self.y.partial_cmp(&other.y),
            ord => ord,
        }
    }
}

// Line segment for sweep-line
// One contour edge on its way into the sweep line, carrying the vertex indices
// the emitted triangles will refer to.
#[derive(Clone, Copy, Debug)]
struct FEdge {
    a: FPoint,
    a_index: u32,
    b: FPoint,
    b_index: u32,
}

impl FEdge {
    fn min_x(&self) -> f32 {
        self.a.x.min(self.b.x)
    }
    fn max_x(&self) -> f32 {
        self.a.x.max(self.b.x)
    }
    fn min_y(&self) -> f32 {
        self.a.y.min(self.b.y)
    }
    fn max_y(&self) -> f32 {
        self.a.y.max(self.b.y)
    }

    fn is_degenerate(&self) -> bool {
        self.a == self.b
    }

    /// How far along this edge `p` lies, for a point already known to be
    /// collinear with it. Measured on the edge's DOMINANT axis, whose span is
    /// guaranteed non-zero for a non-degenerate edge -- the other axis can be
    /// exactly zero (a horizontal or vertical edge) and would divide by it.
    ///
    /// Only ever used to order cuts along an edge, so the absolute value does
    /// not have to be accurate, just monotonic.
    fn param_of(&self, p: FPoint) -> f32 {
        let (dx, dy) = (self.b.x - self.a.x, self.b.y - self.a.y);
        if dx.abs() >= dy.abs() {
            (p.x - self.a.x) / dx
        } else {
            (p.y - self.a.y) / dy
        }
    }

    /// True when `p` -- already known to be collinear with this edge -- lies
    /// strictly between its endpoints, so cutting here yields two real
    /// fragments.
    ///
    /// The exact `p == a || p == b` rejection is what keeps ADJACENT edges of a
    /// contour, and two contours that merely touch at a shared vertex, from
    /// generating cuts: they meet exactly at an endpoint, which is already the
    /// shared endpoint the sweep wants, so there is nothing to split.
    ///
    /// Containment is tested on the dominant axis alone. A bounding-box test
    /// would be wrong for a vertical edge, whose x-extent is a single value that
    /// a collinear-to-within-rounding point can miss by an ulp.
    fn spans_collinear_point(&self, p: FPoint) -> bool {
        if p == self.a || p == self.b {
            return false;
        }
        let (dx, dy) = (self.b.x - self.a.x, self.b.y - self.a.y);
        if dx.abs() >= dy.abs() {
            p.x > self.a.x.min(self.b.x) && p.x < self.a.x.max(self.b.x)
        } else {
            p.y > self.a.y.min(self.b.y) && p.y < self.a.y.max(self.b.y)
        }
    }

    /// The point where this edge's line meets `other`'s, as parameters along
    /// each plus the point itself. Only meaningful once `orient` has confirmed
    /// the two genuinely straddle each other, which also bounds `denom` away
    /// from zero.
    fn crossing_point(&self, other: &FEdge) -> Option<(f32, f32, FPoint)> {
        let (rx, ry) = (self.b.x - self.a.x, self.b.y - self.a.y);
        let (sx, sy) = (other.b.x - other.a.x, other.b.y - other.a.y);
        let denom = rx * sy - ry * sx;
        if denom == 0.0 {
            return None;
        }
        let (qx, qy) = (other.a.x - self.a.x, other.a.y - self.a.y);
        let t = (qx * sy - qy * sx) / denom;
        let u = (qx * ry - qy * rx) / denom;
        // Evaluated on THIS edge only, and then shared with the other edge by
        // the caller, so both fragments meet at bit-identical f32 coordinates.
        Some((t, u, FPoint::new(self.a.x + rx * t, self.a.y + ry * t)))
    }
}

/// Which side of the directed line `a -> b` the point `p` falls on: `1` left,
/// `-1` right, `0` collinear.
///
/// WHY THIS IS NOT A BARE SIGN TEST. The whole planarisation rests on agreeing
/// with itself about whether a point is on a line, and the natural expression
/// `cross(b - a, p - a)` is evaluated in f32 with four subtractions and two
/// multiplications, every one of them rounded. A point that an SVG author placed
/// exactly on an edge does not survive the viewBox transform and curve
/// flattening as an exactly-collinear f32 triple; the computed cross product
/// comes out as some tiny non-zero value whose SIGN IS NOISE. Testing it against
/// literal `0.0` therefore decides T-junctions by coin flip -- measured, before
/// this predicate existed, at 10 of 40 sampled scale/rotation/offset
/// combinations of one notched chevron mis-filling, five of them across ~3% of
/// the shape's area, with no pattern to which ones.
///
/// So instead of comparing against zero, compare against THE ROUNDING ERROR OF
/// THIS VERY EXPRESSION. `bound` is Shewchuk's static filter for `orient2d`
/// (*Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
/// Predicates*, 1997, §4.3): the error in the computed determinant is at most
/// `(3 + 16u)u` times the sum of the magnitudes of its two product terms, where
/// `u` is the unit roundoff -- `2^-24` for f32, which is `f32::EPSILON / 2`.
/// That makes the strict bound `1.5 * f32::EPSILON * (|left| + |right|)`.
///
/// This uses `4.0`, ~2.7x the strict bound, deliberately:
///
///   * The filter is only sound for inputs that are exact f32 values. Ours are
///     not -- they arrive through a viewBox transform, an affine node
///     transform, and Bezier flattening, so they already carry several rounding
///     steps of accumulated error before this predicate sees them. The margin
///     covers that history.
///   * The two outcomes are not symmetric in cost. Wrongly calling a truly
///     collinear point "off the line" is the bug this replaces: it silently
///     skips a T-junction and corrupts every winding region downstream of it.
///     Wrongly calling a very-nearly-collinear point "on the line" merely snaps
///     a cut onto an existing vertex that is within ~5e-7 RELATIVE of where it
///     would otherwise have gone -- far below one device pixel at any scale,
///     because the bound is relative to the operands and therefore scale-free.
///     Erring towards collinear is the safe direction, so the margin goes there.
///
/// Being relative is what makes this usable at all here: the tessellator runs in
/// document units, and those range from a 0-1 unit square to `TUR.svg`'s
/// `0 -30000 90000 60000` viewBox. Any absolute epsilon would be simultaneously
/// far too coarse for one and far too fine for the other.
fn orient(a: FPoint, b: FPoint, p: FPoint) -> i32 {
    let left = (b.x - a.x) * (p.y - a.y);
    let right = (b.y - a.y) * (p.x - a.x);
    let det = left - right;
    let bound = 4.0 * f32::EPSILON * (left.abs() + right.abs());
    if det > bound {
        1
    } else if det < -bound {
        -1
    } else {
        0
    }
}

/// Split every edge wherever it properly crosses another, so the sweep line only
/// ever meets edges at shared endpoints.
///
/// WHY this is needed at all. `SweepTessellator` derives its events solely from
/// the endpoints it is handed: `push_edge` queues exactly two events per edge,
/// and `SweepActiveEdge::split` only fires for a vertex that already exists.
/// Two edges crossing in their interiors therefore produce no event at all. The
/// active-edge list keeps its pre-crossing order past that point, so every
/// `upper_region_winding` accumulated downstream is attributed to the wrong
/// region and `region_is_interior` is asked the wrong question. That is not a
/// fill-rule fault -- it corrupts even-odd exactly as badly -- and it is why a
/// pentagram (five segments that cross, the compact way to draw a five-pointed
/// star, and what `TUR.svg` uses) came out as a single trapezoid. The comment
/// above `FPoint` claiming self-intersecting polygons were handled was simply
/// untrue.
///
/// WHY the split happens here rather than inside the sweep. Both fragments of a
/// cut edge are given the SAME `FPoint` -- the f32 pair is computed once and
/// copied -- and the same newly interned vertex index. The crossing is thus a
/// genuine shared endpoint that `FPoint::eq` matches exactly and `pop_events`
/// merges into one event, which is the representation the rest of the sweep
/// already knows how to handle. Computing the crossing inside the sweep would
/// instead need a freshly derived f32 point to compare exactly `Equal` against
/// both segments in `FSegment::compare_to_point`, an exact predicate on an
/// inexact point; it generally does not, and the edge would be missed again.
///
/// THE THREE WAYS TWO EDGES CAN MEET, all handled here and all decided by
/// `orient` rather than by any comparison against a literal zero:
///
///   * A proper crossing, in both interiors. A new vertex is interned at the
///     intersection and both edges are cut there.
///   * A T-JUNCTION -- one edge's endpoint lying on the other's interior. The
///     other edge is cut AT THAT ALREADY-EXISTING VERTEX; nothing new is
///     interned. This matters because the two then share one index and one
///     bit-identical `FPoint`, so there is no sliver fragment and no chance of
///     the sweep seeing two "almost equal" events. Left to the sweep's own
///     exact predicate this case failed at 10 of 40 sampled scale/rotation/
///     offset combinations of a single notched chevron; see `orient`.
///   * A COLLINEAR OVERLAP. Each edge is cut at whichever of the other's
///     endpoints lies inside it, which reduces the shared stretch to a pair of
///     exactly-coincident edges with identical endpoints -- the form
///     `SweepPendingEdge::splice` already merges, summing their windings so an
///     interior seam between two abutting shapes correctly cancels.
///
/// Two edges that merely share an endpoint -- every adjacent pair in a contour,
/// and two contours touching at a point -- produce nothing, because
/// `spans_collinear_point` rejects exact endpoint equality. They are already in
/// the shared-endpoint form the sweep wants.
///
/// WHAT THIS IS NOT: an outline or boolean pass. No edge is dropped, reversed or
/// merged, so each fragment keeps the direction -- and therefore the signed
/// winding contribution -- of the edge it came from. Deciding what is interior
/// stays entirely with `fill_rule`, and both rules keep their own answer: the
/// pentagram's centre pentagon reaches winding 2, which non-zero fills and
/// even-odd does not. `tests/self_intersecting_fill.rs` pins both.
///
/// COST: sweep-and-prune on x. Edges are visited in min-x order and compared
/// only against edges still open at that x, with a y-overlap reject on top, so
/// for the geometry this actually sees -- short, spatially separated edges from
/// flattened curves, map tiles and chart series -- the active set stays small
/// and the pass is near-linear.
///
/// Measured on a 5000-edge closed outline with no crossings at all -- the shape
/// of the worst realistic input, a dense map tile. Min of 40 runs, re-flattening
/// before each, which matters: `fill()` is NOT idempotent on one `flatten()`, so
/// timing a loop of repeated `fill()` calls measures progressively degenerate
/// work and understates the cost by ~2x. An earlier version of this comment
/// quoted 1.87ms -> 2.31ms from exactly that mistake; the honest figures are
///
///   no planarisation at all      1.58ms   (and self-intersections drawn wrong)
///   crossings only               1.98ms
///   crossings + T-junctions      1.95ms
///
/// so the whole pass costs ~25%, and handling the degenerate meetings on top of
/// proper crossings is free -- the two early-outs on the orientation signs pay
/// for the extra predicates. A deliberately adversarial 200 mutually overlapping
/// pentagrams goes 16ms -> 599ms, but that is not a regression to compare
/// against: the 16ms run emitted 3080 triangles and was simply wrong, the 599ms
/// run emits 435047 and is right. That input has ~218000 genuine crossings and a
/// crossing has to become a vertex.
///
/// It degrades to quadratic only for a path made mostly of long mutually
/// overlapping segments -- which is also the only shape whose crossing COUNT is
/// quadratic, so the pairing is not what dominates there either.
fn split_edges_at_crossings(edges: &mut Vec<FEdge>, verts: &mut Vec<VVertex>, gpu_expand: bool) {
    // Three edges or fewer cannot cross without sharing an endpoint, and a
    // shared endpoint is not a proper crossing.
    if edges.len() < 4 {
        return;
    }

    // Crossing parameters found on each edge, as (t along that edge, vertex).
    let mut cuts: Vec<Vec<(f32, u32)>> = vec![Vec::new(); edges.len()];
    // Interning by exact bit pattern, so a point where three or more edges meet
    // becomes ONE vertex for all of them rather than several coincident ones
    // that the sweep would then treat as separate events.
    let mut interned: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();
    let mut found_any = false;

    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| {
        edges[a]
            .min_x()
            .partial_cmp(&edges[b].min_x())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut active: Vec<usize> = Vec::new();
    for &i in &order {
        let ei = edges[i];
        // A zero-length edge is collinear with everything and spans nothing, so
        // it can neither be cut nor cut anything. `prepare_points` already
        // merges points closer than its `dist_tol`, and `push_edge` drops these
        // outright, but skipping here keeps them out of `orient`, where a null
        // direction vector would make every test read "collinear".
        if ei.is_degenerate() {
            continue;
        }
        let sweep_x = ei.min_x();
        // Anything whose x-extent has been passed can never meet a later edge.
        active.retain(|&j| edges[j].max_x() >= sweep_x);
        for &j in &active {
            let ej = edges[j];
            if ei.min_y() > ej.max_y() || ej.min_y() > ei.max_y() {
                continue;
            }

            // Where each edge's endpoints fall relative to the other's line.
            // These four signs classify every way two segments can meet, and
            // they are the ONLY place the decision is made -- no comparison
            // against a raw zero, and no reliance on the intersection
            // parameters landing exactly on 0.0 or 1.0.
            //
            // Ordered as two early-outs rather than four unconditional tests:
            // if `ej` lies wholly to one side of `ei`'s line they cannot meet,
            // which settles the overwhelming majority of candidate pairs for
            // half the arithmetic. Worth stating because the obvious "compute
            // all four, then decide" costs twice as much on a dense path --
            // measured at 4.12ms against 1.95ms per fill on the 5000-edge
            // benchmark described below.
            let oja = orient(ei.a, ei.b, ej.a);
            let ojb = orient(ei.a, ei.b, ej.b);
            if oja != 0 && oja == ojb {
                continue;
            }
            let oia = orient(ej.a, ej.b, ei.a);
            let oib = orient(ej.a, ej.b, ei.b);
            if oia != 0 && oia == oib {
                continue;
            }

            // An endpoint lying ON the other edge: cut the other edge AT THAT
            // EXISTING VERTEX. This is the T-junction case, and reusing the
            // vertex rather than manufacturing a new one is the whole point --
            // the two edges then meet at a bit-identical `FPoint` with one
            // shared index, which is exactly the shared endpoint the sweep is
            // built to handle. Four independent tests rather than a match,
            // because a collinear overlap satisfies several at once and each
            // contributes its own cut.
            //
            // Gated on some orientation actually being zero so the common pair
            // never builds the table.
            let mut met_at_a_vertex = false;
            if oja == 0 || ojb == 0 || oia == 0 || oib == 0 {
                for (host, host_slot, guest, guest_index, guest_orient) in [
                    (ei, i, ej.a, ej.a_index, oja),
                    (ei, i, ej.b, ej.b_index, ojb),
                    (ej, j, ei.a, ei.a_index, oia),
                    (ej, j, ei.b, ei.b_index, oib),
                ] {
                    if guest_orient == 0 && host.spans_collinear_point(guest) {
                        cuts[host_slot].push((host.param_of(guest), guest_index));
                        found_any = true;
                        met_at_a_vertex = true;
                    }
                }
            }
            // Two straight segments meet in at most one point unless they are
            // collinear, and a collinear pair is fully described by the cuts
            // just made. Either way there is no separate crossing left to find.
            if met_at_a_vertex {
                continue;
            }

            // A proper crossing: each edge's endpoints fall on strictly
            // opposite sides of the other. Because both straddles are strict --
            // every sign is beyond `orient`'s error bound -- the two lines are
            // meaningfully non-parallel here, so the division below is well
            // conditioned rather than merely non-zero.
            if oja * ojb >= 0 || oia * oib >= 0 {
                continue;
            }
            let Some((ti, tj, point)) = ei.crossing_point(&ej) else {
                continue;
            };
            let key = (point.x.to_bits(), point.y.to_bits());
            let index = *interned.entry(key).or_insert_with(|| {
                let index = verts.len() as u32;
                // A crossing point is interior to the fill body, so it carries
                // what every other body vertex carries: u = 0.5 (fully covered,
                // not on the AA fringe). In GPU-expand mode v/stroke_dist hold
                // the outward normal used to push a vertex onto the fringe, and
                // an interior point is never pushed anywhere -- hence zero.
                verts.push(VVertex {
                    x: point.x,
                    y: point.y,
                    u: 0.5,
                    v: if gpu_expand { 0.0 } else { 1.0 },
                    stroke_dist: 0.0,
                    clip_radius: 0.0,
                });
                index
            });
            cuts[i].push((ti, index));
            cuts[j].push((tj, index));
            found_any = true;
        }
        active.push(i);
    }

    if !found_any {
        return;
    }

    let mut split = Vec::with_capacity(edges.len() + interned.len() * 2);
    for (i, edge) in edges.iter().enumerate() {
        if cuts[i].is_empty() {
            split.push(*edge);
            continue;
        }
        // Walk the cuts in order along the edge so the fragments are contiguous
        // and each keeps the original direction.
        cuts[i].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut from = (edge.a, edge.a_index);
        for &(_, index) in &cuts[i] {
            let point = FPoint::new(verts[index as usize].x, verts[index as usize].y);
            // Two cuts can round to the same f32 point on a very short edge;
            // emitting a zero-length fragment there would be dropped by
            // `push_edge` anyway, so skip it and keep walking.
            if point != from.0 {
                split.push(FEdge {
                    a: from.0,
                    a_index: from.1,
                    b: point,
                    b_index: index,
                });
            }
            from = (point, index);
        }
        if edge.b != from.0 {
            split.push(FEdge {
                a: from.0,
                a_index: from.1,
                b: edge.b,
                b_index: edge.b_index,
            });
        }
    }
    *edges = split;
}

#[derive(Clone, Copy, Debug)]
struct FSegment {
    start: FPoint,
    end: FPoint,
}

impl FSegment {
    fn new(start: FPoint, end: FPoint) -> Self {
        Self { start, end }
    }

    // Returns ordering of point relative to segment: Less = right, Greater = left, Equal = on
    fn compare_to_point(&self, p: FPoint) -> std::cmp::Ordering {
        let c =
            (p.x - self.start.x) * (self.end.y - p.y) - (p.y - self.start.y) * (self.end.x - p.x);
        if c > 0.0 {
            std::cmp::Ordering::Greater
        } else if c < 0.0 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

// Running winding accumulation for non-zero fill rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FWinding(i32);

impl FWinding {
    const ZERO: Self = Self(0);
    const POS: Self = Self(1);
    const NEG: Self = Self(-1);
}

impl std::ops::Add for FWinding {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

// Events for the sweep line
#[derive(Clone, Copy, Debug)]
struct SweepEvent {
    vertex: FPoint,
    vertex_index: u32,
    pending_edge: Option<SweepPendingEdge>,
}

impl PartialEq for SweepEvent {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex
    }
}
impl Eq for SweepEvent {}

impl PartialOrd for SweepEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SweepEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior with BinaryHeap (which is max-heap)
        other
            .vertex
            .partial_cmp(&self.vertex)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Clone, Copy, Debug)]
struct SweepPendingEdge {
    winding: FWinding,
    end: FPoint,
    end_index: u32,
}

impl SweepPendingEdge {
    fn to_segment(self, start: FPoint) -> FSegment {
        FSegment::new(start, self.end)
    }

    fn compare(&self, other: &Self, start: FPoint) -> std::cmp::Ordering {
        if self
            .end
            .partial_cmp(&other.end)
            .map_or(false, |o| o != std::cmp::Ordering::Greater)
        {
            other.to_segment(start).compare_to_point(self.end).reverse()
        } else {
            self.to_segment(start).compare_to_point(other.end)
        }
    }

    fn overlaps(&self, other: &Self, start: FPoint) -> bool {
        self.compare(other, start) == std::cmp::Ordering::Equal
    }

    fn splice(&mut self, mut other: Self) -> Option<SweepEvent> {
        if other
            .end
            .partial_cmp(&self.end)
            .map_or(false, |o| o != std::cmp::Ordering::Greater)
        {
            std::mem::swap(self, &mut other);
        }
        self.winding = self.winding + other.winding;
        if self.end == other.end {
            return None;
        }
        Some(SweepEvent {
            vertex: self.end,
            vertex_index: self.end_index,
            pending_edge: Some(other),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct SweepActiveEdge {
    is_temporary: bool,
    winding: FWinding,
    start_index: u32,
    end_index: u32,
    edge: FSegment,
    upper_region_winding: FWinding,
    lower_mono: Option<usize>,
    upper_mono: Option<usize>,
}

impl SweepActiveEdge {
    fn split(&mut self, vertex: FPoint) -> Option<SweepPendingEdge> {
        let end = self.edge.end;
        if vertex == end {
            return None;
        }
        self.edge = FSegment::new(self.edge.start, vertex);
        Some(SweepPendingEdge {
            winding: self.winding,
            end,
            end_index: self.end_index,
        })
    }
}

// Monotone polygon tessellator
#[derive(Clone, Copy, Debug, PartialEq)]
enum MonoSide {
    Lower,
    Upper,
}

#[derive(Clone, Debug)]
struct MonoVertex {
    index: u32,
    pos: FPoint,
}

#[derive(Clone, Debug)]
struct MonoPoly {
    side: MonoSide,
    stack: Vec<MonoVertex>,
}

impl MonoPoly {
    fn new() -> Self {
        Self {
            side: MonoSide::Lower,
            stack: Vec::new(),
        }
    }

    fn start(&mut self, index: u32, pos: FPoint) {
        self.stack.clear();
        self.stack.push(MonoVertex { index, pos });
    }

    fn finish(&mut self, index: u32, out: &mut Vec<u32>) {
        let mut v1 = self.stack.pop().unwrap();
        while let Some(v0) = self.stack.pop() {
            out.push(v0.index);
            out.push(v1.index);
            out.push(index);
            v1 = v0;
        }
    }

    fn push_vertex(&mut self, side: MonoSide, index: u32, pos: FPoint, out: &mut Vec<u32>) {
        if side == self.side {
            let mut v1 = self.stack.pop().unwrap();
            loop {
                let v0 = if let Some(v0) = self.stack.last() {
                    v0.clone()
                } else {
                    break;
                };
                let seg = FSegment::new(v0.pos, pos);
                let cmp = seg.compare_to_point(v1.pos);
                match (cmp, side) {
                    (std::cmp::Ordering::Less, MonoSide::Lower) => break,
                    (std::cmp::Ordering::Equal, _) => break,
                    (std::cmp::Ordering::Greater, MonoSide::Upper) => break,
                    _ => (),
                }
                self.stack.pop();
                out.push(v0.index);
                out.push(v1.index);
                out.push(index);
                v1 = v0;
            }
            self.stack.push(v1);
            self.stack.push(MonoVertex { index, pos });
        } else {
            let vertex = self.stack.pop().unwrap();
            let mut v1 = vertex.clone();
            while let Some(v0) = self.stack.pop() {
                out.push(v0.index);
                out.push(v1.index);
                out.push(index);
                v1 = v0;
            }
            self.stack.push(vertex);
            self.stack.push(MonoVertex { index, pos });
            self.side = side;
        }
    }
}

// Simple arena for monotone polygons
#[derive(Default, Debug)]
struct MonoArena {
    polys: Vec<Option<MonoPoly>>,
    free: Vec<usize>,
    pool: Vec<MonoPoly>,
}

impl MonoArena {
    /// Clear per-fill state, recycling any live polygon stacks into the pool.
    fn reset(&mut self) {
        for slot in self.polys.drain(..) {
            if let Some(poly) = slot {
                self.pool.push(poly);
            }
        }
        self.free.clear();
    }

    fn insert(&mut self, poly: MonoPoly) -> usize {
        if let Some(idx) = self.free.pop() {
            self.polys[idx] = Some(poly);
            idx
        } else {
            let idx = self.polys.len();
            self.polys.push(Some(poly));
            idx
        }
    }

    fn remove(&mut self, idx: usize) -> MonoPoly {
        let poly = self.polys[idx].take().unwrap();
        self.free.push(idx);
        poly
    }

    fn get_mut(&mut self, idx: usize) -> &mut MonoPoly {
        self.polys[idx].as_mut().unwrap()
    }

    fn start_mono(&mut self, index: u32, pos: FPoint) -> usize {
        let mut poly = self.pool.pop().unwrap_or_else(MonoPoly::new);
        poly.start(index, pos);
        self.insert(poly)
    }

    fn finish_mono(&mut self, idx: usize, index: u32, out: &mut Vec<u32>) {
        let mut poly = self.remove(idx);
        poly.finish(index, out);
        self.pool.push(poly);
    }
}

// The main sweep-line tessellator.
// Reusable across fills: `reset()` clears per-fill state but keeps all
// buffer capacities (events, active edges, monotone stacks).
#[derive(Default, Debug)]
struct SweepTessellator {
    fill_rule: FillRule,
    active_edges: Vec<SweepActiveEdge>,
    // Initial edge events are batched here by push_edge, then sorted once at
    // tessellate time and consumed through `events_cursor`. One sort of a
    // flat vec beats 2E BinaryHeap pushes + pops by a large margin (heap pop
    // was ~40% of fill CPU on map fixtures).
    events: Vec<SweepEvent>,
    events_cursor: usize,
    // Dynamically discovered events (splices of overlapping collinear
    // edges) — rare, so a small side heap merged with the sorted stream.
    event_queue: std::collections::BinaryHeap<SweepEvent>,
    // Scratch for create_right_edges (avoids a per-event Vec collect).
    right_edges_scratch: Vec<SweepActiveEdge>,
    pending_scratch: Vec<SweepPendingEdge>,
    left_edges_scratch: Vec<SweepActiveEdge>,
    mono_arena: MonoArena,
}

impl SweepTessellator {
    fn reset(&mut self, fill_rule: FillRule) {
        self.fill_rule = fill_rule;
        self.active_edges.clear();
        self.events.clear();
        self.events_cursor = 0;
        self.event_queue.clear();
        self.right_edges_scratch.clear();
        self.mono_arena.reset();
    }

    /// Queue the two events for a single edge.
    ///
    /// The per-ring `push_ring` below is the fast path and covers every fill
    /// whose contours do not cross themselves. This one exists for the
    /// planarised edge set produced by `split_edges_at_crossings`, whose
    /// fragments no longer form contiguous rings.
    fn push_edge(&mut self, start: FPoint, start_index: u32, end: FPoint, end_index: u32) {
        if start == end {
            return;
        }
        let less = start.partial_cmp(&end) == Some(std::cmp::Ordering::Less);
        // Even-odd only ever counts parity, so a flipped edge keeps POS;
        // non-zero has to record that the edge was reversed into sort order.
        let (start, start_index, end, end_index, winding) = match self.fill_rule {
            FillRule::NonZero => {
                if less {
                    (start, start_index, end, end_index, FWinding::POS)
                } else {
                    (end, end_index, start, start_index, FWinding::NEG)
                }
            }
            FillRule::EvenOdd => {
                if less {
                    (start, start_index, end, end_index, FWinding::POS)
                } else {
                    (end, end_index, start, start_index, FWinding::POS)
                }
            }
        };
        self.events.push(SweepEvent {
            vertex: start,
            vertex_index: start_index,
            pending_edge: Some(SweepPendingEdge {
                winding,
                end,
                end_index,
            }),
        });
        self.events.push(SweepEvent {
            vertex: end,
            vertex_index: end_index,
            pending_edge: None,
        });
    }

    /// Queue events for one closed ring of `count` vertices whose positions
    /// are `ring[k]` and whose output indices are `base_index + k`.
    ///
    /// Equivalent to pushing every ring edge as a (pending@min-end,
    /// plain@max-end) event pair, but emitted per VERTEX so the redundant
    /// plain event of pass-through vertices (which already get a pending
    /// event with the same index at the same position) is skipped: ~V
    /// events instead of 2V, which halves the event sort.
    fn push_ring(&mut self, ring: &[VVertex], base_index: u32) {
        use std::cmp::Ordering;
        let count = ring.len();
        let fp = |k: usize| FPoint::new(ring[k].x, ring[k].y);
        // Winding of an edge that had to be flipped into sort order.
        let flipped_winding = match self.fill_rule {
            FillRule::NonZero => FWinding::NEG,
            FillRule::EvenOdd => FWinding::POS,
        };
        for i in 0..count {
            let h = (i + count - 1) % count;
            let j = (i + 1) % count;
            let p = fp(i);
            let ph = fp(h);
            let pj = fp(j);
            let index = base_index + i as u32;
            let has_in = ph != p;
            let has_out = p != pj;
            // Incoming edge (h -> i): pending sits at i when i is not the
            // sort-order start (exactly push_edge's flip branch).
            let in_pending = has_in && ph.partial_cmp(&p) != Some(Ordering::Less);
            // Outgoing edge (i -> j): pending sits at i when i is the start.
            let out_pending = has_out && p.partial_cmp(&pj) == Some(Ordering::Less);
            if !in_pending && !out_pending {
                if has_in || has_out {
                    // Local maximum (or half-degenerate): the vertex still
                    // has to appear in the event stream.
                    self.events.push(SweepEvent {
                        vertex: p,
                        vertex_index: index,
                        pending_edge: None,
                    });
                }
                continue;
            }
            // Emit pendings in the same relative order the per-edge scheme
            // produced after a stable sort: incoming (queued by edge h)
            // before outgoing (queued by edge i) — except vertex 0, whose
            // outgoing edge 0 was queued before its incoming edge count-1.
            let in_event = in_pending.then(|| SweepEvent {
                vertex: p,
                vertex_index: index,
                pending_edge: Some(SweepPendingEdge {
                    winding: flipped_winding,
                    end: ph,
                    end_index: base_index + h as u32,
                }),
            });
            let out_event = out_pending.then(|| SweepEvent {
                vertex: p,
                vertex_index: index,
                pending_edge: Some(SweepPendingEdge {
                    winding: FWinding::POS,
                    end: pj,
                    end_index: base_index + j as u32,
                }),
            });
            let (first, second) = if i == 0 {
                (out_event, in_event)
            } else {
                (in_event, out_event)
            };
            if let Some(ev) = first {
                self.events.push(ev);
            }
            if let Some(ev) = second {
                self.events.push(ev);
            }
        }
    }

    /// Sweep all pushed edges, appending triangle indices to `out`.
    fn tessellate_into(&mut self, out: &mut Vec<u32>) {
        // Sort the batched events into sweep order (ascending vertex).
        self.events.sort_unstable_by(|a, b| {
            a.vertex
                .partial_cmp(&b.vertex)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.events_cursor = 0;

        let mut pending = std::mem::take(&mut self.pending_scratch);
        let mut left_edges = std::mem::take(&mut self.left_edges_scratch);
        pending.clear();
        left_edges.clear();

        while let Some((vertex, vi)) = self.pop_events(&mut pending) {
            self.handle_vertex(vertex, vi, &mut pending, &mut left_edges, out);
            pending.clear();
            left_edges.clear();
        }
        self.pending_scratch = pending;
        self.left_edges_scratch = left_edges;
    }

    /// Peek the next event across the sorted stream and the dynamic heap.
    fn peek_event(&self) -> Option<&SweepEvent> {
        let stream = self.events.get(self.events_cursor);
        let heap = self.event_queue.peek();
        match (stream, heap) {
            (Some(s), Some(h)) => {
                // Prefer the sorted stream on ties (stable order).
                if h.vertex.partial_cmp(&s.vertex) == Some(std::cmp::Ordering::Less) {
                    Some(h)
                } else {
                    Some(s)
                }
            }
            (Some(s), None) => Some(s),
            (None, h) => h,
        }
    }

    /// Pop the next event across the sorted stream and the dynamic heap.
    fn next_event(&mut self) -> Option<SweepEvent> {
        let take_heap = match (self.events.get(self.events_cursor), self.event_queue.peek()) {
            (Some(s), Some(h)) => h.vertex.partial_cmp(&s.vertex) == Some(std::cmp::Ordering::Less),
            (Some(_), None) => false,
            (None, Some(_)) => true,
            (None, None) => return None,
        };
        if take_heap {
            self.event_queue.pop()
        } else {
            let e = self.events[self.events_cursor];
            self.events_cursor += 1;
            Some(e)
        }
    }

    fn pop_events(&mut self, pending: &mut Vec<SweepPendingEdge>) -> Option<(FPoint, u32)> {
        let event = self.next_event()?;
        let mut vertex_index = event.vertex_index;
        if let Some(pe) = event.pending_edge {
            pending.push(pe);
        }
        while let Some(next) = self.peek_event() {
            if next.vertex != event.vertex {
                break;
            }
            let next = self.next_event().unwrap();
            vertex_index = vertex_index.min(next.vertex_index);
            if let Some(pe) = next.pending_edge {
                pending.push(pe);
            }
        }
        Some((event.vertex, vertex_index))
    }

    fn handle_vertex(
        &mut self,
        vertex: FPoint,
        vi: u32,
        pending: &mut Vec<SweepPendingEdge>,
        left_edges: &mut Vec<SweepActiveEdge>,
        out: &mut Vec<u32>,
    ) {
        let mut incident_range = self.find_incident_range(vertex);
        self.fix_temporary_edges(vertex, &mut incident_range);
        let incident_start = incident_range.start;

        // Remove incident edges and collect split pending edges
        for mut ae in self.active_edges.drain(incident_range.clone()) {
            if let Some(pe) = ae.split(vertex) {
                pending.push(pe);
            }
            left_edges.push(ae);
        }

        // Sort and splice pending edges
        pending.sort_by(|a, b| a.compare(b, vertex));
        let mut write = 0;
        for read in 1..pending.len() {
            let pe1 = pending[read];
            if pending[write].overlaps(&pe1, vertex) {
                if let Some(ev) = pending[write].splice(pe1) {
                    self.event_queue.push(ev);
                }
            } else {
                write += 1;
                pending[write] = pe1;
            }
        }
        if !pending.is_empty() {
            pending.truncate(write + 1);
        }

        // Determine lower/upper monotone polygons from left edges
        let (lower_mono, upper_mono) = if left_edges.is_empty() {
            self.connect_left_vertex(incident_start)
        } else {
            self.finish_left_monos(vi, left_edges, out)
        };

        if let Some(lm) = lower_mono {
            self.mono_arena
                .get_mut(lm)
                .push_vertex(MonoSide::Upper, vi, vertex, out);
        }
        if let Some(um) = upper_mono {
            self.mono_arena
                .get_mut(um)
                .push_vertex(MonoSide::Lower, vi, vertex, out);
        }

        if pending.is_empty() {
            self.connect_right_vertex(vi, vertex, incident_start, lower_mono, upper_mono);
        } else {
            self.create_right_edges(vi, vertex, incident_start, pending, lower_mono, upper_mono);
        }
    }

    fn find_incident_range(&self, vertex: FPoint) -> std::ops::Range<usize> {
        // NOTE: this must stay a full linear scan. The active list is only
        // approximately ordered: crossing sliver edges (self-intersecting or
        // degenerate input) leave locally swapped entries, and a binary
        // search then misses incident edges (verified on real map fixtures).
        let start = self
            .active_edges
            .iter()
            .position(|ae| ae.edge.compare_to_point(vertex) != std::cmp::Ordering::Less)
            .unwrap_or(self.active_edges.len());
        let end = self
            .active_edges
            .iter()
            .rposition(|ae| ae.edge.compare_to_point(vertex) != std::cmp::Ordering::Greater)
            .map_or(0, |i| i + 1);
        start..end
    }

    fn fix_temporary_edges(&mut self, vertex: FPoint, range: &mut std::ops::Range<usize>) {
        while range.start > 0 && self.active_edges[range.start - 1].is_temporary {
            range.start -= 1;
            self.active_edges[range.start].split(vertex);
        }
        while range.end < self.active_edges.len() && self.active_edges[range.end].is_temporary {
            self.active_edges[range.end].split(vertex);
            range.end += 1;
        }
    }

    fn last_lower_winding(&self, incident_start: usize) -> FWinding {
        if incident_start == 0 {
            FWinding::ZERO
        } else {
            self.active_edges[incident_start - 1].upper_region_winding
        }
    }

    fn region_is_interior(&self, winding: FWinding) -> bool {
        match self.fill_rule {
            FillRule::NonZero => winding.0 != 0,
            FillRule::EvenOdd => (winding.0 & 1) != 0,
        }
    }

    fn connect_left_vertex(&mut self, incident_start: usize) -> (Option<usize>, Option<usize>) {
        if !self.region_is_interior(self.last_lower_winding(incident_start)) {
            return (None, None);
        }
        let ae0 = self.active_edges[incident_start - 1];
        let ae1 = self.active_edges[incident_start];
        if ae0.edge.start.partial_cmp(&ae1.edge.start) != Some(std::cmp::Ordering::Greater) {
            let um = self.mono_arena.start_mono(ae1.start_index, ae1.edge.start);
            let old = self.active_edges[incident_start].lower_mono.replace(um);
            (old, Some(um))
        } else {
            let lm = self.mono_arena.start_mono(ae0.start_index, ae0.edge.start);
            let old = self.active_edges[incident_start - 1].upper_mono.replace(lm);
            (Some(lm), old)
        }
    }

    fn finish_left_monos(
        &mut self,
        vi: u32,
        left_edges: &[SweepActiveEdge],
        out: &mut Vec<u32>,
    ) -> (Option<usize>, Option<usize>) {
        for le in &left_edges[..left_edges.len() - 1] {
            if self.region_is_interior(le.upper_region_winding) {
                if let Some(um) = le.upper_mono {
                    self.mono_arena.finish_mono(um, vi, out);
                }
            }
        }
        (
            left_edges.first().unwrap().lower_mono,
            left_edges.last().unwrap().upper_mono,
        )
    }

    fn connect_right_vertex(
        &mut self,
        vi: u32,
        vertex: FPoint,
        incident_start: usize,
        lower_mono: Option<usize>,
        upper_mono: Option<usize>,
    ) {
        let lower_winding = self.last_lower_winding(incident_start);
        if !self.region_is_interior(lower_winding) {
            return;
        }
        let end_point = {
            let ae0 = self.active_edges[incident_start - 1];
            let ae1 = self.active_edges[incident_start];
            if ae0.edge.end.partial_cmp(&ae1.edge.end) != Some(std::cmp::Ordering::Greater) {
                (ae0.edge.end, ae0.end_index)
            } else {
                (ae1.edge.end, ae1.end_index)
            }
        };
        self.active_edges.insert(
            incident_start,
            SweepActiveEdge {
                is_temporary: true,
                winding: FWinding::ZERO,
                start_index: vi,
                end_index: end_point.1,
                edge: FSegment::new(vertex, end_point.0),
                upper_region_winding: lower_winding,
                lower_mono,
                upper_mono,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn create_right_edges(
        &mut self,
        vi: u32,
        vertex: FPoint,
        incident_start: usize,
        pending: &[SweepPendingEdge],
        mut lower_mono: Option<usize>,
        upper_mono: Option<usize>,
    ) {
        let mut lower_winding = self.last_lower_winding(incident_start);
        let mut new_edges = std::mem::take(&mut self.right_edges_scratch);
        new_edges.clear();
        for (i, pe) in pending.iter().enumerate() {
            let upper_winding = lower_winding + pe.winding;
            let um = if self.region_is_interior(upper_winding) {
                if i == pending.len() - 1 {
                    upper_mono
                } else {
                    Some(self.mono_arena.start_mono(vi, vertex))
                }
            } else {
                None
            };
            new_edges.push(SweepActiveEdge {
                is_temporary: false,
                winding: pe.winding,
                start_index: vi,
                end_index: pe.end_index,
                edge: pe.to_segment(vertex),
                upper_region_winding: upper_winding,
                lower_mono,
                upper_mono: um,
            });
            lower_winding = upper_winding;
            lower_mono = um;
        }
        self.active_edges
            .splice(incident_start..incident_start, new_edges.drain(..));
        self.right_edges_scratch = new_edges;
    }
}

#[cfg(test)]
mod stroke_tests {
    use super::*;

    fn straight_stroke(cap: LineCap) -> (Vec<VVertex>, Vec<u32>) {
        let mut path = VectorPath::new();
        path.move_to(0.0, 0.0);
        path.line_to(20.0, 0.0);
        let mut tess = Tessellator::default();
        tess.flatten(&path, 0.25);
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        tess.stroke(10.0, cap, LineJoin::Miter, 4.0, 1.0, &mut verts, &mut indices);
        (verts, indices)
    }

    #[test]
    fn round_caps_emit_one_carrier_pair_per_end() {
        let (verts, indices) = straight_stroke(LineCap::Round);

        // Two body pairs plus one two-vertex carrier at each end: no fan.
        assert_eq!(verts.len(), 2 * 2 + 2 * 2);
        assert_eq!(indices.len(), 3 * 6);
        assert_eq!(
            verts.iter().map(|vertex| vertex.v).collect::<Vec<_>>(),
            [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
        );
        assert!(verts
            .iter()
            .all(|vertex| !(vertex.x == 0.0 && vertex.y == 0.0)
                && !(vertex.x == 20.0 && vertex.y == 0.0)));
        assert!(indices.iter().all(|&index| index < verts.len() as u32));
    }

    #[test]
    fn butt_start_emits_no_dead_feather_vertices() {
        let (verts, indices) = straight_stroke(LineCap::Butt);
        let mut referenced = vec![false; verts.len()];
        for &index in &indices {
            referenced[index as usize] = true;
        }

        // Start contributes only its structural pair; the indexed end
        // carrier remains necessary to place the longitudinal AA ramp.
        assert_eq!(verts.len(), 2 + 4);
        assert!(referenced.into_iter().all(|is_referenced| is_referenced));
    }
}

// ---- Fixture replay benchmark ----------------------------------------------
// Replays fill() inputs captured via MAKEPAD_TESS_DUMP (see fixture_dump).
// Run:
//   TESS_FIXTURE=/path/a.tessfix[,b.tessfix...] \
//   cargo test -p makepad-svg --release fixture_bench -- --ignored --nocapture
// Env knobs:
//   TESS_REPS=N          timing repetitions (default 3, best-of for totals)
//   TESS_TRUST=1         enable trust_fill_winding (map fast path)
//   TESS_NO_BBOX=1       disable probe bbox prefilter (original baseline)
//   TESS_GOLDEN_WRITE=p  write per-record output summary to p
//   TESS_GOLDEN_CHECK=p  compare outputs against a written summary
#[cfg(test)]
mod fixture_bench {
    use super::*;

    struct FixContour {
        closed: bool,
        has_explicit_winding: bool,
        winding: Winding,
        pts: Vec<(f32, f32, u8)>,
    }

    struct FixRecord {
        aa: f32,
        line_join: LineJoin,
        miter_limit: f32,
        gpu_expand_fill: bool,
        contours: Vec<FixContour>,
    }

    struct Cursor<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn u32(&mut self) -> u32 {
            let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
            self.pos += 4;
            v
        }
        fn f32(&mut self) -> f32 {
            f32::from_bits(self.u32())
        }
        fn done(&self) -> bool {
            self.pos >= self.data.len()
        }
    }

    fn parse_fixture(data: &[u8]) -> Vec<FixRecord> {
        let mut cur = Cursor { data, pos: 0 };
        let mut out = Vec::new();
        while !cur.done() {
            assert_eq!(cur.u32(), 0x5445_5346, "bad magic at {}", cur.pos - 4);
            assert_eq!(cur.u32(), 1, "bad version");
            let aa = cur.f32();
            let line_join = match cur.u32() {
                0 => LineJoin::Miter,
                1 => LineJoin::Round,
                _ => LineJoin::Bevel,
            };
            let miter_limit = cur.f32();
            let gpu_expand_fill = cur.u32() != 0;
            let n_paths = cur.u32() as usize;
            let mut contours = Vec::with_capacity(n_paths);
            for _ in 0..n_paths {
                let count = cur.u32() as usize;
                let flags = cur.u32();
                let mut pts = Vec::with_capacity(count);
                for _ in 0..count {
                    let x = cur.f32();
                    let y = cur.f32();
                    let f = cur.u32() as u8;
                    pts.push((x, y, f));
                }
                contours.push(FixContour {
                    closed: flags & 1 != 0,
                    has_explicit_winding: flags & 2 != 0,
                    winding: if flags & 4 != 0 { Winding::CW } else { Winding::CCW },
                    pts,
                });
            }
            out.push(FixRecord {
                aa,
                line_join,
                miter_limit,
                gpu_expand_fill,
                contours,
            });
        }
        out
    }

    /// Load a record's flattened contours into the tessellator, bit-exact,
    /// recomputing only dx/dy/len (winding fixup + dedup already happened
    /// before the dump; re-running them could diverge).
    fn load_record(tess: &mut Tessellator, rec: &FixRecord) {
        tess.points.clear();
        tess.paths.clear();
        for c in &rec.contours {
            let first = tess.points.len();
            tess.paths.push(SubPath {
                first,
                count: c.pts.len(),
                closed: c.closed,
                winding: c.winding,
                has_explicit_winding: c.has_explicit_winding,
                convex: false,
                nbevel: 0,
            });
            for &(x, y, flags) in &c.pts {
                tess.points.push(VPoint {
                    x,
                    y,
                    flags,
                    ..Default::default()
                });
            }
        }
        // Same direction/length pass as prepare_points.
        for i in 0..tess.paths.len() {
            let first = tess.paths[i].first;
            let count = tess.paths[i].count;
            if count < 2 {
                continue;
            }
            for j in 0..count {
                let j1 = if j + 1 < count { j + 1 } else { 0 };
                let p0 = tess.points[first + j];
                let p1 = tess.points[first + j1];
                let mut dx = p1.x - p0.x;
                let mut dy = p1.y - p0.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-6 {
                    let il = 1.0 / len;
                    dx *= il;
                    dy *= il;
                }
                tess.points[first + j].dx = dx;
                tess.points[first + j].dy = dy;
                tess.points[first + j].len = len;
            }
        }
    }

    fn fnv1a64(hash: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *hash ^= b as u64;
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }

    struct RecordResult {
        n_verts: usize,
        n_indices: usize,
        area: f64,
        hash: u64,
        nan_verts: usize,
    }

    fn summarize(verts: &[VVertex], indices: &[u32]) -> RecordResult {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut nan_verts = 0usize;
        for v in verts {
            for f in [v.x, v.y, v.u, v.v, v.stroke_dist] {
                if !f.is_finite() {
                    nan_verts += 1;
                }
                fnv1a64(&mut hash, &f.to_bits().to_le_bytes());
            }
        }
        for i in indices {
            fnv1a64(&mut hash, &i.to_le_bytes());
        }
        let mut area = 0.0f64;
        for tri in indices.chunks_exact(3) {
            let a = &verts[tri[0] as usize];
            let b = &verts[tri[1] as usize];
            let c = &verts[tri[2] as usize];
            area += 0.5
                * ((b.x as f64 - a.x as f64) * (c.y as f64 - a.y as f64)
                    - (c.x as f64 - a.x as f64) * (b.y as f64 - a.y as f64));
        }
        RecordResult {
            n_verts: verts.len(),
            n_indices: indices.len(),
            area,
            hash,
            nan_verts,
        }
    }

    /// Recompute the baseline probe's per-contour fill-side signs for a
    /// record (diagnostics for trusted-winding divergences).
    fn probe_signs_for(rec: &FixRecord) -> Vec<f32> {
        let mut tess = Tessellator::default();
        load_record(&mut tess, rec);
        let mut valid_paths: Vec<(usize, usize, f32)> = Vec::new();
        for sp in &tess.paths {
            if sp.count >= 3 {
                valid_paths.push((sp.first, sp.count, 1.0));
            }
        }
        let fill_rule = if tess
            .paths
            .iter()
            .any(|sp| sp.count >= 3 && sp.has_explicit_winding)
        {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        };
        let mut out = Vec::new();
        for i in 0..valid_paths.len() {
            let (first, count, _) = valid_paths[i];
            let mut sign = 1.0f32;
            let mut found = false;
            for j in 0..count {
                let j1 = (j + 1) % count;
                let p0 = tess.points[first + j];
                let p1 = tess.points[first + j1];
                let mut ex = p1.x - p0.x;
                let mut ey = p1.y - p0.y;
                let e2 = ex * ex + ey * ey;
                if e2 <= 1e-12 {
                    continue;
                }
                let inv_e = 1.0 / e2.sqrt();
                ex *= inv_e;
                ey *= inv_e;
                let nx = ey;
                let ny = -ex;
                let mx = (p0.x + p1.x) * 0.5;
                let my = (p0.y + p1.y) * 0.5;
                let local_len = p0.len.max(p1.len);
                let base_eps = (local_len * 1e-3).max(1e-4);
                for s in [1.0f32, 4.0, 16.0] {
                    let eps = base_eps * s;
                    let plus_filled = point_in_fill_rule(
                        mx + nx * eps,
                        my + ny * eps,
                        &tess.points,
                        &valid_paths,
                        None,
                        fill_rule,
                    );
                    let minus_filled = point_in_fill_rule(
                        mx - nx * eps,
                        my - ny * eps,
                        &tess.points,
                        &valid_paths,
                        None,
                        fill_rule,
                    );
                    if plus_filled != minus_filled {
                        sign = if plus_filled { 1.0 } else { -1.0 };
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                let area = poly_area(&tess.points[first..first + count]);
                sign = if area >= 0.0 { 1.0 } else { -1.0 };
            }
            out.push(sign);
        }
        out
    }

    fn percentile(sorted: &[u64], p: f64) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[idx]
    }

    #[test]
    #[ignore]
    fn fixture_bench() {
        let fixtures = std::env::var("TESS_FIXTURE")
            .expect("set TESS_FIXTURE=/path/one.tessfix[,two.tessfix...]");
        let reps: usize = std::env::var("TESS_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let trust = std::env::var("TESS_TRUST").map_or(false, |v| v == "1");
        let no_bbox = std::env::var("TESS_NO_BBOX").map_or(false, |v| v == "1");
        let golden_write = std::env::var("TESS_GOLDEN_WRITE").ok();
        let golden_check = std::env::var("TESS_GOLDEN_CHECK").ok();

        for fixture_path in fixtures.split(',') {
            let data = std::fs::read(fixture_path).expect(fixture_path);
            let records = parse_fixture(&data);
            let n_contours: usize = records.iter().map(|r| r.contours.len()).sum();
            let n_pts: usize = records
                .iter()
                .flat_map(|r| r.contours.iter().map(|c| c.pts.len()))
                .sum();
            println!(
                "\n== {fixture_path}: {} fills, {} contours, {} points (trust={} bbox={})",
                records.len(),
                n_contours,
                n_pts,
                trust,
                !no_bbox
            );

            let mut tess = Tessellator::default();
            tess.set_trust_fill_winding(trust);
            tess.disable_probe_bbox = no_bbox;
            let mut verts = Vec::new();
            let mut indices = Vec::new();

            // Stage split rep (with stats), then clean timing reps.
            let mut stats = FillStats::default();
            let mut results = Vec::with_capacity(records.len());
            for rec in &records {
                load_record(&mut tess, rec);
                tess.fill_impl(
                    rec.aa,
                    rec.line_join,
                    rec.miter_limit,
                    rec.gpu_expand_fill,
                    &mut verts,
                    &mut indices,
                    Some(&mut stats),
                );
                results.push(summarize(&verts, &indices));
            }

            let mut best_total_ns = u64::MAX;
            let mut per_fill_ns: Vec<u64> = vec![0; records.len()];
            for _ in 0..reps.max(1) {
                let mut total = 0u64;
                for (ri, rec) in records.iter().enumerate() {
                    load_record(&mut tess, rec);
                    let t0 = std::time::Instant::now();
                    tess.fill_impl(
                        rec.aa,
                        rec.line_join,
                        rec.miter_limit,
                        rec.gpu_expand_fill,
                        &mut verts,
                        &mut indices,
                        None,
                    );
                    let ns = t0.elapsed().as_nanos() as u64;
                    per_fill_ns[ri] = ns;
                    total += ns;
                }
                best_total_ns = best_total_ns.min(total);
            }

            let stage_total = (stats.joins_ns
                + stats.sign_ns
                + stats.body_emit_ns
                + stats.sweep_ns
                + stats.fringe_ns)
                .max(1);
            println!(
                "stage split (1 rep): joins {:.1}ms {:.0}% | sign {:.1}ms {:.0}% | body {:.1}ms {:.0}% | sweep {:.1}ms {:.0}% | fringe {:.1}ms {:.0}%",
                stats.joins_ns as f64 / 1e6,
                stats.joins_ns as f64 / stage_total as f64 * 100.0,
                stats.sign_ns as f64 / 1e6,
                stats.sign_ns as f64 / stage_total as f64 * 100.0,
                stats.body_emit_ns as f64 / 1e6,
                stats.body_emit_ns as f64 / stage_total as f64 * 100.0,
                stats.sweep_ns as f64 / 1e6,
                stats.sweep_ns as f64 / stage_total as f64 * 100.0,
                stats.fringe_ns as f64 / 1e6,
                stats.fringe_ns as f64 / stage_total as f64 * 100.0,
            );
            let mut sorted = per_fill_ns.clone();
            sorted.sort_unstable();
            println!(
                "total {:.1}ms (best of {reps}) | per-fill p50 {:.1}us p95 {:.1}us max {:.1}ms",
                best_total_ns as f64 / 1e6,
                percentile(&sorted, 0.50) as f64 / 1e3,
                percentile(&sorted, 0.95) as f64 / 1e3,
                percentile(&sorted, 1.0) as f64 / 1e6,
            );
            let nan_total: usize = results.iter().map(|r| r.nan_verts).sum();
            if nan_total > 0 {
                println!("WARNING: {} non-finite vertex floats", nan_total);
            }

            let fixture_name = std::path::Path::new(fixture_path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            if let Some(dir) = &golden_write {
                let mut out = String::new();
                for (i, r) in results.iter().enumerate() {
                    out.push_str(&format!(
                        "{} {} {} {:.9e} {:016x}\n",
                        i, r.n_verts, r.n_indices, r.area, r.hash
                    ));
                }
                let path = format!("{dir}/{fixture_name}.golden");
                std::fs::write(&path, out).unwrap();
                println!("golden written: {path}");
            }
            if let Some(dir) = &golden_check {
                let path = format!("{dir}/{fixture_name}.golden");
                let golden = std::fs::read_to_string(&path).expect(&path);
                let mut mismatch_verts = 0usize;
                let mut mismatch_idx = 0usize;
                let mut mismatch_area = 0usize;
                let mut hash_identical = 0usize;
                let mut checked = 0usize;
                for (line, r) in golden.lines().zip(&results) {
                    let mut it = line.split_whitespace();
                    let _i: usize = it.next().unwrap().parse().unwrap();
                    let nv: usize = it.next().unwrap().parse().unwrap();
                    let ni: usize = it.next().unwrap().parse().unwrap();
                    let area: f64 = it.next().unwrap().parse().unwrap();
                    let hash = u64::from_str_radix(it.next().unwrap(), 16).unwrap();
                    checked += 1;
                    if nv != r.n_verts {
                        mismatch_verts += 1;
                        if mismatch_verts <= 5 {
                            println!(
                                "  VERTS DIFF at record {}: golden {} got {}",
                                checked - 1,
                                nv,
                                r.n_verts
                            );
                        }
                    }
                    if ni != r.n_indices {
                        mismatch_idx += 1;
                    }
                    let denom = area.abs().max(1e-6);
                    if ((area - r.area).abs() / denom) > 1e-3 {
                        mismatch_area += 1;
                        if mismatch_area <= 5 {
                            println!(
                                "  AREA DIFF at record {}: golden {:.6} got {:.6}",
                                checked - 1,
                                area,
                                r.area
                            );
                            let rec = &records[checked - 1];
                            println!(
                                "    aa={} join={:?} contours={}",
                                rec.aa,
                                rec.line_join,
                                rec.contours.len()
                            );
                            let probe_signs = probe_signs_for(rec);
                            for (ci, c) in rec.contours.iter().enumerate().take(8) {
                                let pts: Vec<VPoint> = c
                                    .pts
                                    .iter()
                                    .map(|&(x, y, _)| VPoint {
                                        x,
                                        y,
                                        ..Default::default()
                                    })
                                    .collect();
                                println!(
                                    "    contour {ci}: {} pts area {:.4} probe_sign {} closed={} first=({:.2},{:.2})",
                                    c.pts.len(),
                                    poly_area(&pts),
                                    probe_signs[ci],
                                    c.closed,
                                    c.pts[0].0,
                                    c.pts[0].1
                                );
                            }
                        }
                    }
                    if hash == r.hash {
                        hash_identical += 1;
                    }
                }
                assert_eq!(checked, results.len(), "golden record count mismatch");
                println!(
                    "golden check: {}/{} bit-identical, {} vert-count diffs, {} index-count diffs, {} area diffs",
                    hash_identical, checked, mismatch_verts, mismatch_idx, mismatch_area
                );
                // TESS_STRICT=0 downgrades divergence to a report (used to
                // survey documented trusted-winding divergences on glitch
                // geometry without aborting the remaining fixtures).
                let strict = std::env::var("TESS_STRICT").map_or(true, |v| v != "0");
                if strict {
                    assert_eq!(mismatch_verts, 0, "fill body vertex count diverged");
                    assert_eq!(mismatch_area, 0, "triangle area diverged");
                }
            }
        }
    }
}
