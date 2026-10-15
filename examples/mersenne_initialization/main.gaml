model mersenne_initialization
global {
    init { rng <- "mersenne"; seed <- 17; create walker number: 8; }
}
species walker {
    int choice <- rnd(9);
    float value <- rnd(1.0);
    point location <- {0, 0};
    init {
        location <- {rnd(100.0), rnd(100.0)};
        if flip(0.5) { choice <- choice + rnd(3); }
    }
    reflex change { value <- value + rnd(1.0); }
}
experiment validation type: gui {}
