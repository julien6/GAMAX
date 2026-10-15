model import_overrides
import "base.gaml"
global {
    int value <- 3;
    init {signature <- 2; do compute; create Walker number:1;}
    action hook {signature <- signature*10+3;}
}
experiment validation type:gui {}
