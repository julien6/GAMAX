// Bounded continuous flock with synchronous position/velocity snapshots.
model boids

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create bird number: 24; }
    reflex capture {
        ticks <- ticks + 1;
        ask bird { ox <- px; oy <- py; ovx <- vx; ovy <- vy; }
    }
}
species bird {
    float px <- float(index mod 6) + 5.0;
    float py <- float(index div 6) + 5.0;
    float vx <- 0.1;
    float vy <- 0.1;
    float ox <- px;
    float oy <- py;
    float ovx <- vx;
    float ovy <- vy;
    int neighbors <- 0;
    // @initialization
    reflex flock {
        neighbors <- length(bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 16.0));
        if neighbors > 0 {
            vx <- ovx + (mean((bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 16.0)) collect each.ovx) - ovx) * 0.1;
            vy <- ovy + (mean((bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 16.0)) collect each.ovy) - ovy) * 0.1;
            vx <- vx + (mean((bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 16.0)) collect each.ox) - ox) * 0.01;
            vy <- vy + (mean((bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 16.0)) collect each.oy) - oy) * 0.01;
        }
        vx <- vx + sum((bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 1.0)) collect ((ox - each.ox) / max(0.01, (ox - each.ox) * (ox - each.ox) + (oy - each.oy) * (oy - each.oy)))) * 0.02;
        vy <- vy + sum((bird where (each.index != index and (each.ox - ox) * (each.ox - ox) + (each.oy - oy) * (each.oy - oy) < 1.0)) collect ((oy - each.oy) / max(0.01, (ox - each.ox) * (ox - each.ox) + (oy - each.oy) * (oy - each.oy)))) * 0.02;
        vx <- max(-1.0, min(1.0, vx)); vy <- max(-1.0, min(1.0, vy));
        px <- max(0.0, min(20.0, ox + vx)); py <- max(0.0, min(20.0, oy + vy));
    }
}
experiment validation type: gui {}
