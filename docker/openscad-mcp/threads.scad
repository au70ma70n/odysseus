// threads.scad — ISO-style thread library for OpenSCAD MCP Server.
// Usage:  include <threads.scad>;
//
// Main modules:
//   external_thread(d, pitch, length, starts=1, fn=48)
//   internal_thread(d, pitch, length, starts=1, fn=48)
//   threaded_rod(d, pitch, length, starts=1, fn=48)
//   threaded_hole(d, pitch, length, starts=1, fn=48)
//
// d      = nominal (major) diameter of the thread
// pitch  = axial distance per revolution
// length = total thread length
// starts = number of thread starts (1 for normal, 2+ for multi-start)
// fn     = circle segments (higher = smoother)

// Internal clearance added to female threads so male/female parts mate.
_thread_clearance = 0.4;

module _thread_helix(d, pitch, length, starts=1, fn=48) {
    $fn = fn;
    h        = pitch * 0.6;
    r_major  = d / 2;
    r_root   = r_major - h;
    turns    = length / pitch;
    segments = ceil(turns * fn);
    step_a   = 360 * turns / segments;
    step_z   = length / segments;

    tooth_w  = pitch * 0.3;

    for (s = [0 : starts - 1]) {
        sa = s * (360 / starts);
        for (i = [0 : segments - 1]) {
            a1 = sa + i       * step_a;
            a2 = sa + (i + 1) * step_a;
            z1 = i       * step_z;
            z2 = (i + 1) * step_z;

            hull() {
                translate([r_root * cos(a1), r_root * sin(a1), z1])
                    rotate([0, 0, a1])
                    rotate([0, 90, 0])
                    cylinder(h = h, r1 = tooth_w, r2 = 0.05, $fn = 4);
                translate([r_root * cos(a2), r_root * sin(a2), z2])
                    rotate([0, 0, a2])
                    rotate([0, 90, 0])
                    cylinder(h = h, r1 = tooth_w, r2 = 0.05, $fn = 4);
            }
        }
    }
}

// External (male) thread: core cylinder + helical teeth flush with core.
module external_thread(d, pitch, length, starts=1, fn=48) {
    $fn = fn;
    h       = pitch * 0.6;
    minor_d = d - 2 * h;
    union() {
        cylinder(h = length, d = minor_d);
        _thread_helix(d = d, pitch = pitch, length = length,
                      starts = starts, fn = fn);
    }
}

// Internal (female) thread — place inside a difference() block.
// Creates a bore with helical thread grooves sized to accept the
// matching external_thread with printing clearance.
module internal_thread(d, pitch, length, starts=1, fn=48) {
    $fn = fn;
    c       = _thread_clearance;
    h       = pitch * 0.6;
    minor_d = d - 2 * h + c;
    union() {
        cylinder(h = length, d = minor_d);
        _thread_helix(d = d + c, pitch = pitch, length = length,
                      starts = starts, fn = fn);
    }
}

// Convenience aliases.
module threaded_rod(d, pitch, length, starts=1, fn=48) {
    external_thread(d = d, pitch = pitch, length = length,
                    starts = starts, fn = fn);
}

module threaded_hole(d, pitch, length, starts=1, fn=48) {
    internal_thread(d = d, pitch = pitch, length = length,
                    starts = starts, fn = fn);
}
