model dynamic_random_bounds
global {
    int groups <- 3 min:1;
    int lower <- -4;
    int upper <- 7;
    float start <- -1.25;
    float stop <- 3.75;
    init {create Walker number:3;}
}
species Walker {
    int color_index <- rnd(groups-1);
    int integer_sample;
    float float_sample;
    bool branch;
    reflex choose {
        branch <- flip(0.5);
        if branch {integer_sample <- rnd(lower,upper);}
        float_sample <- rnd(start,stop);
        color_index <- rnd(groups-1);
    }
}
experiment validation type:gui {}
