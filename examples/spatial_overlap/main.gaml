model spatial_overlap

global {
    int radius <- 2 min: 1 max: 3;
    geometry shape <- square(50);
    int walkers <- 30;
    init {
        create WalkerHost number: walkers { color <- #red; }
        create WalkerHost number: 0;
        write "Population: " + walkers;
    }
}
grid cell width: 4 height: 3 neighbors: 4 use_regular_agents: false frequency: 0 {
    rgb color <- #white;
    list<cell> nearby <- self neighbors_at radius;
}
species WalkerHost {
    rgb color <- rgb(46,204,113);
    cell place;
    int neighbor_sum;
    init {
        place <- one_of(cell as list);
        location <- place.location;
    }
    reflex update {
        int occupied <- 0;
        loop candidate over: ((place.nearby + place) accumulate (WalkerHost overlapping each)) {
            if (candidate.color = #red) { occupied <- occupied + 1; }
        }
        neighbor_sum <- occupied;
        place <- one_of(place.nearby);
        location <- place.location;
    }
}
experiment validation type: gui {}
