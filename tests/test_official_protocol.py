import pytest
from benchmarks.official_protocol import campaign_schedule,demographic_configuration,frame_cycles


def test_full_schedule_is_paired_reproducible_and_small_first():
    rows=campaign_schedule(['sir','schelling','boids','traffic','predator','ants'])
    assert len(rows)==21600
    assert len({r['id'] for r in rows})==21600
    pairs={}
    for row in rows:pairs.setdefault(row['pair_id'],set()).add(row['engine'])
    assert all(engines=={'gama','jax'} for engines in pairs.values())
    assert rows==campaign_schedule(['sir','schelling','boids','traffic','predator','ants'])
    assert [(r['scale'],r['horizon']) for r in rows]==sorted((r['scale'],r['horizon']) for r in rows)
    assert all(r['status']=='planned_fidelity_gate_required' and r['timeout_seconds']==1800 for r in rows)
    assert frame_cycles(100)==list(range(0,101,10))


def test_scaling_keeps_ecology_and_applies_declared_constraints():
    native={'hosts':495,'infected':5,'beta':0.05}
    assert demographic_configuration(native,['hosts','infected'],0.5)=={'hosts':248,'infected':3,'beta':0.05}
    assert demographic_configuration(native,['hosts'],2,bounds={'hosts':(1,600)})['hosts']==600
    assert native['hosts']==495
    with pytest.raises(ValueError):demographic_configuration(native,['missing'],1)
