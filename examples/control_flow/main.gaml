model control_flow

global {
    int total <- 0;
    float sampled <- 0.0;
    int code <- 0;
    init { create walker number: 4; rng <- "java"; seed <- 17; }
    action accumulate_sample(int amount, float sample) {
        total <- total + amount;
        sampled <- sampled + sample;
    }
    action encode_draws(int first_draw, int second_draw) {
        code <- first_draw * 10 + second_draw;
    }
    reflex nested {
        loop outer from: 0 to: 1 {
            loop inner from: 3 to: -1 step: -2 {
                int amount <- inner + outer;
                if inner > 1 { do accumulate_sample sample: rnd(1.0) amount: amount; }
                else if inner = 1 { total <- total + 10; }
                else { total <- total - 1; }
            }
        }
        loop times: 2 { total <- total + 1; }
        loop times: 0 { sampled <- rnd(1.0); }
        do encode_draws second_draw: rnd(9) first_draw: rnd(9);
    }
}
species walker {
    int count <- 0;
    reflex progress {
        loop k from: 0 to: 2 { count <- count + k; }
    }
}
experiment validation type: gui {}
