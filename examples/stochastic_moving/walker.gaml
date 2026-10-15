model moving_actions
global { int ticks <- 0; reflex count { ticks <- ticks + 1; } }
species walker skills: [moving] {
    float displacement <- 0.0;
    init {
        location <- {95.0, 5.0 + index * 10.0};
        speed <- 2.0;
        heading <- index * 45.0;
    }
    action turn(int angle) { heading <- heading + angle; }
    reflex walk when: speed > 0.0 and flip(0.8) {
        point previous <- location;
        if flip(0.2) { do turn(rnd(0, 359)); }
        do move;
        displacement <- previous distance_to location;
    }
}
