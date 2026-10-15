model creation_batches

global {
    int first_count <- 2;
    int second_count <- 3;
    int total;
    init {
        rng <- "mersenne";
        seed <- 17;
        create walker number: first_count { value <- value + rnd(9); group <- 1; }
        create walker number: second_count { value <- value + rnd(9); group <- 2; }
        create dormant number: 0;
    }
    reflex summarize { total <- sum(walker collect each.value); }
}

species walker {
    point location <- {0,0};
    int group;
    int value <- rnd(9);
    int seen;
    init { seen <- rnd(9); }
    reflex update { value <- value + 1; }
}

species dormant { int value; reflex unused { value <- value + 1; } }

experiment validation type: gui {}
