model mutable_agent_lists

global {
    list<Walker> cache;
    Walker selected;
    int length_now;
    init {
        create Walker number:3;
        cache <- Walker as list;
        cache << one_of(cache);
        cache >> one_of(cache);
    }
    reflex rotate {
        selected <- one_of(cache);
        cache >> selected;
        cache << selected;
        length_now <- length(cache);
    }
}
species Walker {int value <- index;}
experiment validation type:gui {}
