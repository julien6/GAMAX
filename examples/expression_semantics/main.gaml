model expression_semantics

global {
    int denominator <- 2;
    int successes <- 0;
    int division <- 0;
    int remainder <- 0;
    float quotient <- 0.0;
    int choice <- 0;
    point center <- {0,0};
    point total <- {0,0};
    point scaled <- {0,0};
    bool disabled <- false;
    init {
        create particle number: 4;
        rng <- "mersenne";
        seed <- 17;
    }
    reflex expressions {
        successes <- particle count (each.value > 1);
        center <- mean(particle collect each.position);
        total <- sum(particle collect each.position);
        scaled <- center / denominator;
        quotient <- 9 / denominator;
        division <- -9 div denominator;
        remainder <- -9 mod denominator;
        choice <- !disabled ? rnd(9) : rnd(9);
        quotient <- disabled ? 1 / zero : quotient;
    }
    int zero;
}

species particle {
    int value <- index;
    point position <- {index * 2.0, index * 4.0};
}

experiment validation type: gui {}
