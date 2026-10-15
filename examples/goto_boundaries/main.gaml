model goto_boundaries
global {init {create Walker number:4;}}
species Walker skills:[moving] {
    point destination <- {0,0};
    init {location <- {50,50};destination <- {50,110};if index=0 {destination <- {-10,50};} if index=1 {destination <- {110,50};} if index=2 {destination <- {50,-10};}}
    reflex travel {do goto target:destination speed:15.0;}
}
experiment validation type:gui {}
