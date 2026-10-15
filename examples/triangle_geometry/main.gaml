model triangle_geometry
global {init {create Walker number:2;}}
species Walker skills:[moving] {
    init {shape <- triangle(15);}
    reflex travel {do wander amplitude:45.0 speed:2.0;}
}
experiment validation type:gui {}
