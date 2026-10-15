"""Performance eligibility must include translation CPU and rejected campaigns."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    'amortized_analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_amortized_performance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rejected_campaign_has_no_performance_analysis(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps({
        'performance_valid': False, 'reason': 'unbounded RPC retention'}))
    (tmp_path / 'measurements.json').write_text('[]')
    with pytest.raises(ValueError, match='unbounded RPC retention'):
        module.analyze(tmp_path)
    assert not (tmp_path / 'analysis.json').exists()


def test_translation_cpu_regression_blocks_joint_gain(tmp_path):
    ratios = {'total_including_translation_seconds': 2., 'cpu_seconds': 2.,
              'cpu_including_translation_seconds': .5, 'peak_pss_bytes': 2.}
    run = {'seed': 1, 'eligible': True, 'gama_over_jax': {
        'total_seconds': 2., 'cpu_seconds': 2., 'compute_seconds': 2.}}
    summary = {'complete': True, 'completed': 2, 'planned': 2, 'pairs': [{
        'case': 'fixture', 'mode': 'compute', 'eligible': True,
        'runs': [run], 'batch_gama_over_jax': ratios}]}
    measurements = []
    for engine, seconds in [('gama', 2.), ('jax', 1.)]:
        measurements.append({'case': 'fixture', 'mode': 'compute', 'engine': engine,
            'total_including_translation_seconds': seconds,
            'batch': {'reports': [{'seed': 1, 'total_seconds': seconds}]}})
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    (tmp_path / 'measurements.json').write_text(json.dumps(measurements))
    result = module.analyze(tmp_path)
    assert result['reports'][0]['batch_joint_gain_observed'] is False
    assert result['reports'][0]['seed_statistics']['total_seconds']['ci95'] is None
