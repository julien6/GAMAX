model grid_cast_sort_nil
global {
    geometry shape <- rectangle(6,4);
    init { create Walker number:2 with:(location:{2.5,1.5}); }
}
grid Cell width:6 height:4 neighbors:8 {
    float distance <- topology(Cell) distance_between [self,{2,2}];
    int value <- index mod 3;
}
species Walker skills:[moving] {
    int own;
    int ordered;
    int nearby;
    int phase;
    reflex lookup {
        Cell cell <- Cell(location);
        own <- cell.index;
        list<Cell> places <- cell.neighbors;
        nearby <- length(places);
        places <- places where (each.value>0) sort_by each.value;
        ordered <- last(places).index;
        point absent <- point(last(Cell where (each.value<0)));
        location <- absent;
        if absent=nil {phase <- phase+1;}
    }
}
experiment validation type:gui {}
