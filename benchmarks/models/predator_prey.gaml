// Bounded spatial predator-prey ABM: explicit active-slot birth/death and grass regrowth.
model predator_prey

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create patch number: 64; create prey number: 32; create predator number: 12; }
    reflex capture {
        ticks <- ticks + 1;
        ask prey { ox <- px; oy <- py; old_active <- active; old_energy <- energy; }
        ask predator { ox <- px; oy <- py; old_active <- active; old_energy <- energy; }
        ask patch {
            previous_food <- food;
            food <- max(0, min(5, food + 1 - length(prey where (each.old_active and each.ox = px and each.oy = py))));
        }
    }
}
species patch {
    int px <- index mod 8;
    int py <- index div 8;
    int food <- 5;
    int previous_food <- 5;
}
species prey {
    int px <- index mod 8;
    int py <- index div 8;
    int ox <- px;
    int oy <- py;
    int energy <- 4;
    int old_energy <- energy;
    bool active <- index < 16;
    bool old_active <- active;
    int threats <- 0;
    // @initialization
    reflex live {
        threats <- length(predator where (each.old_active and each.ox = ox and each.oy = oy));
        if old_active {
            energy <- energy - 1 + min(2, sum((patch where (each.px = ox and each.py = oy)) collect each.previous_food));
            if threats > 0 or energy <= 0 { active <- false; }
            else {
                px <- (ox + 1 + ((index + ticks + scenario) mod 2)) mod 8;
                py <- (oy + ((index + ticks) mod 2)) mod 8;
            }
        }
        else {
            if length(prey where (each.old_active and each.old_energy >= 6 and abs(each.ox - ox) + abs(each.oy - oy) <= 1)) > 0 {
                active <- true; energy <- 3;
            }
        }
    }
}
species predator {
    int px <- (index * 3) mod 8;
    int py <- (index * 5) mod 8;
    int ox <- px;
    int oy <- py;
    int energy <- 6;
    int old_energy <- energy;
    bool active <- index < 4;
    bool old_active <- active;
    int meals <- 0;
    // @initialization
    reflex hunt {
        meals <- length(prey where (each.old_active and each.ox = ox and each.oy = oy));
        if old_active {
            energy <- energy - 1 + meals * 4;
            if energy <= 0 { active <- false; }
            else {
                px <- (ox + 1) mod 8;
                py <- (oy + ((index + ticks + scenario) mod 2)) mod 8;
            }
        }
        else {
            if length(predator where (each.old_active and each.old_energy >= 10 and abs(each.ox - ox) + abs(each.oy - oy) <= 1)) > 0 {
                active <- true; energy <- 5;
            }
        }
    }
}
experiment validation type: gui {}
