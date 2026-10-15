// Authored bounded grid ABM. Explicit scalar coordinates, synchronous updates.
model forest_fire

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create patch number: 64; }
    reflex capture {
        ticks <- ticks + 1;
        ask patch { previous <- status; }
    }
}
species patch {
    int px <- index mod 8;
    int py <- index div 8;
    int status <- 1;
    int previous <- 1;
    int burning_neighbors <- 0;
    // @initialization
    reflex spread {
        burning_neighbors <- length(patch where ((abs(each.px - px) + abs(each.py - py)) = 1 and each.previous = 2));
        if previous = 2 { status <- 0; }
        else {
            if previous = 1 and burning_neighbors > 0 { status <- 2; }
        }
    }
}
experiment validation type: gui {}
