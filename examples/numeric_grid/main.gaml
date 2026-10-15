model numeric_grid

global torus: false {
    int next_draw;
    int tick;
    init { next_draw <- rnd(1000000); }
    reflex advance { tick <- tick + 1; }
}

grid cell width: 4 height: 3 neighbors: 4 use_regular_agents: false frequency: 1 {
    int degree <- length(self neighbors_at 2);
    int sample <- rnd(9);
    int dynamic_degree;
    reflex update {
        sample <- sample + 1;
        dynamic_degree <- length(self neighbors_at 1);
    }
}

experiment validation type: gui {}
