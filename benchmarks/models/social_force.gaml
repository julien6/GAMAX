// Social-force-inspired evacuation with pairwise repulsion, desired motion and a door.
model social_force

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create pedestrian number: 24; }
    reflex capture { ticks <- ticks + 1; ask pedestrian { ox <- px; oy <- py; } }
}
species pedestrian {
    float px <- float(index mod 6) + 10.0;
    float py <- float(index div 6) + 8.0;
    float ox <- px;
    float oy <- py;
    float vx <- 0.0;
    float vy <- 0.0;
    bool evacuated <- false;
    // @initialization
    reflex move when: not evacuated {
        vx <- vx + (1.0 - vx) * 0.2;
        vy <- vy + ((10.0 - oy) * 0.2 - vy) * 0.2;
        vx <- vx + sum((pedestrian where (each.index != index and not each.evacuated and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 1.0)) collect ((ox - each.ox) / max(0.01, (ox - each.ox) * (ox - each.ox) + (oy - each.oy) * (oy - each.oy)))) * 0.05;
        vy <- vy + sum((pedestrian where (each.index != index and not each.evacuated and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 1.0)) collect ((oy - each.oy) / max(0.01, (ox - each.ox) * (ox - each.ox) + (oy - each.oy) * (oy - each.oy)))) * 0.05;
        vx <- max(-2.0, min(2.0, vx)); vy <- max(-2.0, min(2.0, vy));
        px <- ox + vx * 0.2; py <- max(0.0, min(20.0, oy + vy * 0.2));
        if px > 18.0 and (py < 9.0 or py > 11.0) { px <- 18.0; vx <- 0.0; }
        if px > 20.0 { evacuated <- true; }
    }
}
experiment validation type: gui {}
