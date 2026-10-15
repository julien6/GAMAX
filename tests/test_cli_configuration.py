"""Public CLI configuration reaches the same generated native-coupled backend."""
import json
from pathlib import Path
import pytest
from gamax.cli import main
from gamax.validation import compare_traces
ROOT=Path(__file__).resolve().parents[1]


def test_cli_compiles_selected_experiment_and_runs_external_seed(tmp_path):
    generated=tmp_path/'generated';output=tmp_path/'output'
    assert main(['compile',str(ROOT/'examples/grid_species_order/main.gaml'),'--experiment','validation','--capacities','{}','-o',str(generated)])==0
    assert main(['run',str(generated/'model_jax.py'),'--seed','2','--steps','10','--parameters','{}','-o',str(output)])==0
    reference=ROOT/'tests/fixtures/native/grid-species-order-native-coupled-2026-10-13/grid_species_order/2/gama_trace.json'
    manifest=json.loads((generated/'manifest.json').read_text())
    assert compare_traces(json.loads(reference.read_text()),json.loads((output/'jax_trace.json').read_text()),manifest['fields'])['passed']

@pytest.mark.parametrize('command,option',[('compile','--capacities'),('run','--parameters')])
def test_configuration_requires_json_object(tmp_path,command,option):
    with pytest.raises(SystemExit) as error:main([command,'unused',option,'[]','-o',str(tmp_path/'output')])
    assert error.value.code==2
    assert not (tmp_path/'output').exists()
