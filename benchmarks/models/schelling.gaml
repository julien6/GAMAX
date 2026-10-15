// Asynchronous relocation to a deterministic candidate vacant site; Moore neighbors.
model schelling

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create resident number: 32; }
    reflex advance_time { ticks <- ticks + 1; }
}
species resident {
    int px <- index mod 8;
    int py <- index div 8;
    int group_id <- index mod 2;
    int neighbors <- 0;
    int similar <- 0;
    int candidate_x <- 0;
    int candidate_y <- 0;
    bool unhappy <- false;
    // @initialization
    reflex relocate {
        neighbors <- length(resident where (each.index != index and abs(each.px - px) <= 1 and abs(each.py - py) <= 1));
        similar <- length(resident where (each.index != index and abs(each.px - px) <= 1 and abs(each.py - py) <= 1 and each.group_id = group_id));
        unhappy <- neighbors > 0 and similar * 2 < neighbors;
        if unhappy {
            candidate_x <- (index * 7 + ticks * 3 + scenario) mod 8;
            candidate_y <- ((index * 7 + ticks * 3 + scenario) div 8) mod 8;
            if length(resident where (each.px = candidate_x and each.py = candidate_y)) = 0 {
                px <- candidate_x; py <- candidate_y;
            }
        }
    }
}
experiment validation type: gui {}
