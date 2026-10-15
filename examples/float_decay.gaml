model float_decay

global {
    float retention <- 0.97;
    init { create particle number: 16; }
}

species particle {
    float concentration <- 1.0 + index * 0.1;
    float previous <- 0.0;
    bool active <- true;
    reflex remember { previous <- concentration; }
    reflex decay when: active {
        concentration <- concentration * retention + 0.001;
        if concentration < 0.05 { active <- false; }
    }
}
experiment validation type: gui {}
