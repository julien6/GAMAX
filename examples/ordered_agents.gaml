model ordered_agents

global {
    int rounds <- 0;
    init { create worker number: 8; }
    reflex advance { rounds <- rounds + 1; }
}

species worker {
    int value <- index + 1;
    int previous <- 0;
    bool active <- true;
    reflex remember { previous <- value; }
    reflex update when: active {
        value <- value + rounds;
        if value > 30 {
            active <- false;
        } else {
            value <- value + 1;
        }
    }
}
experiment validation type: gui {}
