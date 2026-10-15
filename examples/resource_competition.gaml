// Order-sensitive shared-resource ABM: only three of eight agents can collect per step.
model resource_competition

global {
    int available <- 0;
    int rounds <- 0;
    init { create forager number: 8; }
    reflex replenish {
        available <- 3;
        rounds <- rounds + 1;
    }
}

species forager {
    int collected <- 0;
    int previous <- 0;
    reflex remember { previous <- collected; }
    reflex collect when: available > 0 {
        available <- available - 1;
        collected <- collected + 1;
    }
}
experiment validation type: gui {}
