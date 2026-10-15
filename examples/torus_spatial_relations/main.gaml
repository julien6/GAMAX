model torus_spatial_relations
global torus:true {
    geometry shape <- rectangle(20,10);
    init {create Walker number:8 {location <- {1,1}; if index>2 {location <- {index*2+1,1};} if index=7 {location <- {20,1};}}}
}
grid Cell width:4 height:2 {rgb initial_color <- color;}
species Walker skills:[moving] {
    Cell own_cell;
    Walker first_other;
    int nearby_count;
    int phase;
    list<Walker> nearby -> (Walker inside own_cell);
    reflex find {
        own_cell <- Cell first_with (each overlaps self);
        nearby_count <- length(Walker inside own_cell);
        first_other <- first(Walker inside own_cell);
        do move heading:0.0 speed:3.0;
        phase <- phase+1;
    }
}
experiment validation type:gui {}
