"""The pinned native each binding may be clobbered by a getter iterator."""
import pytest
from gamax.compiler import compile_model
from gamax.ir import CompileError


def test_reject_each_reuse_after_nested_getter_iterator(tmp_path):
    source=tmp_path/'unknown.gaml'
    source.write_text('''model unknown global {int total; init {create Actor number:12;} reflex measure {total <- Actor count (each.near_count >= each.value);}} species Actor {int value <- 0; int near_count -> Actor count (each.value = value);} experiment validation type:gui {}''')
    for mode in ('none','fusion-avx-host','fusion-avx-batch'):
        with pytest.raises(CompileError,match='GAMAX_EACH_SCOPE'):compile_model(source,tmp_path/mode,experiment='validation',optimization=mode)
        assert not (tmp_path/mode).exists()


def test_single_outer_getter_preserves_its_own_agent_names(tmp_path):
    source=tmp_path/'different.gaml'
    source.write_text('''model different global {int total; init {create Actor number:12;} reflex measure {total <- Actor count each.test;}} species Actor {int value <- 0; int near_count -> Actor count (each.value = value); bool test -> near_count >= value;} experiment validation type:gui {}''')
    compile_model(source,tmp_path/'compiled',experiment='validation',optimization='fusion-avx-batch')
