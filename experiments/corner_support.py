"""Experimental small corner-throttle support within the existing speed cap.

This is a policy hypothesis based on public force telemetry, not an identified
AGX tyre/engine model. build_corner_experiment.py embeds it in a standalone agent.
"""
import math

CORNER_EXTRA_THROTTLE_CAP = 0.15
CORNER_LATERAL_BUDGET_FRACTION = 0.90


def corner_throttle(o, throttle, brake, target, offset, road_limit,
                    elapsed, ers_mode, lateral_budget):
    c = o['self']
    if (brake > 0.0 or c['pit_state'] or c['fuel_kg'] < 5.0
            or elapsed < 6.0 or c['speed_mps'] < 5.0
            or target <= c['speed_mps'] or target < road_limit - 1e-6):
        return throttle
    heading = math.atan2(math.sin(c['heading_error']), math.cos(c['heading_error']))
    if (abs(heading) >= 0.10 or abs(c['n'] - offset) >= 0.75
            or c.get('front_sliding', True) or c.get('rear_sliding', True)):
        return throttle
    rows = o['preview']['rows'][:3]
    if len(rows) < 3:
        return throttle
    curves = [abs(row[1]) for row in rows]
    # Constant/opening bends only; a tightening or reversing bend keeps V3's
    # original pedal output. All 24 preview rows still constrain road_limit.
    if (curves[0] < 0.002 or max(curves) > 1.05 * curves[0]
            or any(rows[0][1] * row[1] < 0.0 for row in rows[1:])):
        return throttle
    force = c.get('drive_force_n')
    telemetry = (c.get('drag_n'), c.get('rolling_resistance_n'), c.get('g_lat'))
    if (not isinstance(force, (list, tuple)) or len(force) <= ers_mode
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in telemetry)):
        return throttle
    available = force[ers_mode]
    if (not isinstance(available, (int, float)) or not math.isfinite(available)
            or available <= 1.0 or telemetry[0] < 0.0 or telemetry[1] < 0.0):
        return throttle
    load = max(abs(telemetry[2]) * 9.81, c['speed_mps'] ** 2 * max(curves))
    if load >= CORNER_LATERAL_BUDGET_FRACTION * lateral_budget:
        return throttle
    support = max(0.0, telemetry[0] + telemetry[1]) / available
    support = min(CORNER_EXTRA_THROTTLE_CAP, support)
    # Fade out as the unchanged speed target approaches, rather than forcing
    # a minimum throttle at/above the target or during braking.
    support *= max(0.0, min(1.0, target - c['speed_mps']))
    return min(1.0, throttle + support)
