// Grid foraging: fixed food source, return to nest, pheromone diffusion/evaporation.
model ants

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create patch number: 64; create ant number: 24; }
    reflex environment {
        ticks <- ticks + 1;
        ask patch { old_trail <- trail; }
        ask patch {
            trail <- old_trail * 0.90 + mean((patch where (abs(each.px - px) + abs(each.py - py) = 1)) collect each.old_trail) * 0.05;
            trail <- trail + float(length(ant where (each.px = px and each.py = py and each.carrying))) * 0.2;
        }
    }
}
species patch {
    int px <- index mod 8;
    int py <- index div 8;
    float trail <- 0.0;
    float old_trail <- 0.0;
    bool food <- px = 7 and py = 7;
}
species ant {
    int px <- 0;
    int py <- 0;
    int direction <- 0;
    int next_site <- 0;
    int deliveries <- 0;
    float best_trail <- 0.0;
    bool carrying <- false;
    // @initialization
    reflex forage {
        if carrying {
            if px > 0 { px <- px - 1; }
            else { py <- max(0, py - 1); }
            if px = 0 and py = 0 { carrying <- false; deliveries <- deliveries + 1; }
        }
        else {
            best_trail <- max((patch where (abs(each.px - px) + abs(each.py - py) = 1)) collect each.trail);
            if best_trail > 0.0 {
                next_site <- min((patch where (abs(each.px - px) + abs(each.py - py) = 1 and each.trail >= best_trail)) collect each.index);
                px <- next_site mod 8; py <- next_site div 8;
            }
            else {
                direction <- (index + ticks * 3 + scenario) mod 4;
                if direction = 0 { px <- min(7, px + 1); }
                if direction = 1 { py <- min(7, py + 1); }
                if direction = 2 { px <- max(0, px - 1); }
                if direction = 3 { py <- max(0, py - 1); }
            }
            if length(patch where (each.px = px and each.py = py and each.food)) > 0 { carrying <- true; }
        }
    }
}
experiment validation type: gui {}
