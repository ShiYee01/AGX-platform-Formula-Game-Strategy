"""Generate the standalone corner-throttle experiment from the V3 main agent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT/'agent/agent.py').read_text()
    helper = (ROOT/'experiments/corner_support.py').read_text()
    anchor = '\n\nclass Driver:'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '\n\n'+helper+anchor)
    old = """        target, offset = self.traffic_plan(o, plan_speed(o, self.offset), dt)
        target = min(target, plan_speed(o, offset))"""
    new = """        road_before = plan_speed(o, self.offset)
        target, offset = self.traffic_plan(o, road_before, dt)
        road_limit = min(road_before, plan_speed(o, offset))
        target = min(target, road_limit)"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "        self.diagnostic = dict(target=target, offset=offset, wet_race=self.wet_race)"
    new = """        base_throttle = throttle
        throttle = corner_throttle(o, throttle, brake, target, offset, road_limit,
                                   elapsed, 2 if boost else 1, 10.0 * grip_budget(o))
        self.diagnostic = dict(target=target, offset=offset, wet_race=self.wet_race,
                               corner_extra_throttle=throttle - base_throttle)"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    folder = ROOT/'experiments/v3-corner'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'agent.py').write_text('# V3 CORNER EXPERIMENT: test separately from the traffic-only main build.\n'+source)
    (folder/'design.json').write_bytes((ROOT/'agent/design.json').read_bytes())
    print(folder)


if __name__ == '__main__':
    build()
