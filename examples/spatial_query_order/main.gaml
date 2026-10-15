model spatial_query_order
global {
    int walkers <- 110;
    float radius <- 15.0;
    init {create Walker number:walkers;}
}
species Walker {
    int signature;
    int count;
    bool move_now;
    init {location <- {(index mod 11)*8.0+2.0,int(index/11)*8.0+2.0};}
    reflex query {
        signature <- 0;
        count <- 0;
        loop other over:(Walker at_distance radius) {
            signature <- signature*31+other.index;
            count <- count+1;
        }
        move_now <- flip(0.5);
        if move_now {location <- {rnd(99),rnd(99)};}
    }
}
experiment validation type:gui {}
