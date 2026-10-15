import copy
import math

import pytest

from scripts.analyze_robustness import METRICS, analyze


def measurements():
    # Variable absolute costs, constant engine ratio: CI must collapse to that ratio.
    return {'passed': True, 'measurements': [
        dict(engine=engine, mode=mode, repetition=r,
             **{metric: (r + 1) * scale for metric in METRICS})
        for mode in ('compute', 'render') for r in range(4)
        for engine, scale in (('gama', 2), ('jax', 1))
    ]}


def test_constant_paired_ratio_and_reproducibility():
    summary = measurements()
    result = analyze(summary, draws=100)
    assert result == analyze(summary, draws=100)
    for mode in result['results'].values():
        assert mode['joint_improvement_pairs'] == 4
        for metric in mode['metrics'].values():
            assert metric['geometric_mean_ratio'] == pytest.approx(2)
            assert metric['ci95'] == pytest.approx([2, 2])


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'zero', 'nan', 'empty_mode'])
def test_invalid_campaign_cannot_produce_intervals(corruption):
    summary = copy.deepcopy(measurements())
    rows = summary['measurements']
    if corruption == 'missing': rows.pop()
    elif corruption == 'duplicate': rows.append(copy.deepcopy(rows[-1]))
    elif corruption == 'empty_mode': summary['measurements'] = [r for r in rows if r['mode'] == 'compute']
    else: rows[0][METRICS[0]] = 0 if corruption == 'zero' else math.nan
    with pytest.raises(ValueError): analyze(summary, draws=100)


def test_mixed_metric_outcomes_do_not_claim_joint_improvement():
    summary = measurements()
    for row in summary['measurements']:
        if row['engine'] == 'jax': row[METRICS[2]] *= 3
    result = analyze(summary, draws=100)
    for mode in result['results'].values():
        assert mode['joint_improvement_pairs'] == 0
        assert not mode['all_three_marginal_lower_bounds_above_one']
