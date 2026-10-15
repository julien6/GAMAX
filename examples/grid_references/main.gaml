model grid_references

global {
    geometry shape <- square(50);
    init { create WalkerHost number: 3; }
}

grid cell width: 4 height: 3 neighbors: 4 use_regular_agents: false frequency: 0 {
    int degree <- length(self neighbors_at 2);
}

species WalkerHost {
    cell place;
    init {
        place <- one_of(cell as list);
        location <- place.location;
    }
    reflex update {
        place <- one_of((cell as list) where (each.index > 1));
        location <- place.location;
    }
}

experiment validation type: gui {}
