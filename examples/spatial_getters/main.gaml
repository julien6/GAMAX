model spatial_getters
global {
    int walkers <- 110;
    int happy <- 0 update: Walker count each.is_happy;
    int total <- 0 update: sum(Walker collect each.total_nearby);
    init {create Walker number:walkers;}
}
species Walker {
    int group <- rnd(1);
    list<Walker> nearby -> Walker at_distance 15.0;
    int same_group -> nearby count (each.group=group);
    int total_nearby -> length(nearby);
    bool is_happy -> same_group>=0.5*total_nearby;
    reflex migrate when: !is_happy {location <- {rnd(99),rnd(99)};}
}
experiment validation type:gui {}
