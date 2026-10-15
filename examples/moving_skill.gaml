model native_moving
global {
    init { create walker number: 8; }
}
species walker skills: [moving] {
    float displacement <- 0.0;
    init {
        location <- {95.0, 5.0 + index * 10.0};
        speed <- 2.0;
        heading <- index * 45.0;
    }
    reflex walk {
        point previous <- location;
        do move;
        displacement <- previous distance_to location;
    }
}
experiment validation type: gui {}
