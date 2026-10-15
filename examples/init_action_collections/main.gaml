model init_action_collections
global {
    list<cell> available;
    float density <- 0.5;
    int people;
    init {
        do describe;
        do places;
        people <- int(length(available)*density);
        do populate;
    }
    action describe {write "Initializing agents";}
    action places {available <- shuffle(cell);}
    action populate {create Walker number:people;}
}
grid cell width:3 height:2 use_regular_agents:false frequency:0 {}
species Walker {
    cell place;
    init {place <- one_of(available); location <- place.location; available >> place;}
}
experiment validation type:gui {}
