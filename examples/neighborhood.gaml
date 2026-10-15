model neighborhood

global {
    int ticks <- 0;
    init { create patch number: 16; }
    reflex snapshot {
        ticks <- ticks + 1;
        ask patch { previous <- value; }
    }
}
species patch {
    int px <- index mod 4;
    int py <- index div 4;
    int value <- index mod 3;
    int previous <- value;
    int neighbors <- 0;
    float average <- 0.0;
    reflex update {
        neighbors <- length(patch where ((abs(each.px - px) + abs(each.py - py)) = 1));
        average <- mean((patch where ((abs(each.px - px) + abs(each.py - py)) = 1)) collect each.previous);
        value <- sum((patch where ((abs(each.px - px) + abs(each.py - py)) = 1)) collect each.previous) mod 3;
    }
}
experiment validation type: gui {}
