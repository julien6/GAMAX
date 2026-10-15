model persistent_neighbors

global {
    int radius <- 2 min: 1 max: 3;
    geometry shape <- square(50);
    int walkers <- 3;
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
        neighbor_sum <- 0;
        loop candidate over: place.nearby {
            neighbor_sum <- neighbor_sum + candidate.index;
        }
        place <- one_of(place.nearby);
        location <- place.location;
    }
}
experiment validation type: gui {}
