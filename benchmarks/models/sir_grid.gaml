// Deterministic contact transmission; stochastic initialization frozen per scenario.
model sir_grid

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create person number: 64; }
    reflex capture { ticks <- ticks + 1; ask person { previous <- status; } }
}
species person {
    int px <- index mod 8;
    int py <- index div 8;
    int status <- 0;
    int previous <- 0;
    int infected_contacts <- 0;
    int infection_age <- 0;
    // @initialization
    reflex disease {
        infected_contacts <- length(person where ((abs(each.px - px) + abs(each.py - py)) = 1 and each.previous = 1));
        if previous = 0 and infected_contacts > 0 { status <- 1; infection_age <- 0; }
        if previous = 1 {
            infection_age <- infection_age + 1;
            if infection_age >= 6 { status <- 2; }
        }
    }
}
experiment validation type: gui {}
