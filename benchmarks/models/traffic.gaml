// Deterministic parallel-update ring-road car following, bounded speed, no random braking.
model traffic

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create vehicle number: 24; }
    reflex capture { ticks <- ticks + 1; ask vehicle { old_position <- position; } }
}
species vehicle {
    int position <- index * 5;
    int old_position <- position;
    int velocity <- 1;
    int gap <- 0;
    // @initialization
    reflex move {
        gap <- min((vehicle where (each.index != index)) collect ((each.old_position - old_position + 128) mod 128));
        velocity <- min(5, min(velocity + 1, max(0, gap - 1)));
        position <- (old_position + velocity) mod 128;
    }
}
experiment validation type: gui {}
