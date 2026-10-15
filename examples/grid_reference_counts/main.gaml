model grid_reference_counts

global {
    geometry shape <- square(40);
    int radius <- 1 min: 1 max: 2;
    cell anchor;
    int after_queries;
    init { anchor <- one_of(cell as list); create Actor number: 6; }
    reflex lifecycle {
        if cycle = 2 { create Actor number: 2; }
        if cycle = 4 { ask Actor where (each.index = 0) { do die; } }
        after_queries <- rnd(1000);
    }
}

grid cell width: 4 height: 3 neighbors: 4 frequency: 0 {
    list<cell> nearby <- self neighbors_at radius;
}

species Actor {
    cell place;
    bool flagged <- flip(0.5);
    int total;
    int selected;
    int after_count;
    init { place <- one_of(cell as list); location <- place.location; }
    reflex count {
        if cycle mod 4 = 0 or cycle mod 4 = 3 {
            place <- one_of(cell as list);
            location <- place.location;
        }
        if cycle mod 4 = 1 { location <- location + {0.1, 0.0}; }
        if cycle mod 4 = 2 { place <- nil; }
        int n <- 2147483646;
        int k <- 0;
        loop candidate over: ((anchor.nearby + anchor + anchor) accumulate (Actor overlapping each)) {
            n <- n + 1;
            if candidate.flagged { k <- k + 1; }
        }
        total <- n;
        selected <- k;
        after_count <- rnd(1000);
    }
}

experiment validation type: gui { output {} }
