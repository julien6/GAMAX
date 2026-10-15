model dynamic_loops
global {
    int limit <- 3;
    int stride <- -2;
    int total <- 0;
    float sampled <- 0.0;
    init { create walker number: 4; rng <- "mersenne"; seed <- 17; }
    reflex evaluate {
        loop times: limit { total <- total + 1; sampled <- sampled + rnd(1.0); }
        loop k from: limit to: -1 step: stride {
            if k > 0 { total <- total + k; }
        }
        loop i from: 0 to: limit {
            total <- total + i;
            limit <- limit + 1;
        }
        limit <- 3;
        if false { loop times: limit { sampled <- rnd(1.0); } }
    }
}
species walker {
    int count <- 0;
    int repetitions <- 2;
    init { loop times: repetitions { count <- count + 1; } }
    reflex grow { loop k from: 0 to: repetitions { count <- count + k; } }
}
experiment validation type: gui {}
