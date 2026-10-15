model native_collection_selection
global {
    geometry shape <- square(12);
    Walker lead;
    init {create Walker number:3; lead <- first(Walker); ask lead {marker <- 777;}}
}
grid cell width:3 height:2 neighbors:4 use_regular_agents:false frequency:0 {int value <- index;}
species Walker {
    cell direct <- one_of(cell);
    cell from_list <- one_of(cell as list);
    Walker first_other;
    Walker last_other;
    Walker random_other;
    point old_position <- {0,0};
    int marker;
    reflex sample {
        list<Walker> others <- (Walker as list)-self;
        first_other <- first(others);
        last_other <- last(others);
        random_other <- one_of(Walker);
        direct <- one_of(cell);
        from_list <- one_of(cell as list);
        old_position <- copy(location);
        location <- {rnd(11),rnd(11)};
        marker <- location.x as int;
    }
}
experiment validation type:gui {}
