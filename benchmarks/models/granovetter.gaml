// Irreversible threshold adoption on a frozen undirected small-world network.
model granovetter

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create person number: 32; }
    reflex capture { ticks <- ticks + 1; ask person { previous <- adopted; } }
}
species person {
    int link_a <- (index + 1) mod 32;
    int link_b <- (index + 2) mod 32;
    float threshold <- 0.5;
    bool adopted <- false;
    bool previous <- false;
    int contacts <- 0;
    int adopters <- 0;
    // @initialization
    reflex cascade {
        contacts <- length(person where (each.index = link_a or each.index = link_b or each.link_a = index or each.link_b = index));
        adopters <- length(person where ((each.index = link_a or each.index = link_b or each.link_a = index or each.link_b = index) and each.previous));
        if contacts > 0 and adopters / max(1, contacts) >= threshold { adopted <- true; }
    }
}
experiment validation type: gui {}
