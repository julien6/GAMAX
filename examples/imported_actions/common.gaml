model shared_actions
global { int budget <- 100; }
species walker {
    int ticks <- 0;
    float value <- 0.0;
    action advance(float amount) {
        float previous <- value;
        value <- value + amount;
        ticks <- ticks + 1;
        if previous > 0.0 { budget <- budget - 1; }
    }
    reflex evolve {
        if flip(0.7) { do advance(rnd(0.0, 1.0)); }
        if false and flip(0.5) { value <- 1000.0; }
        if true or flip(0.5) { ticks <- ticks + rnd(0, 3); }
    }
}
