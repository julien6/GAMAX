model grid_geometry_distance
global {
    geometry shape <- rectangle(6,4);
    int count;
    init {count <- length(Cell where(each distance_to {2,2}<1.5));}
}
grid Cell width:6 height:4 neighbors:8 {
    float geometric <- self distance_to {2,2};
    float topological <- topology(Cell) distance_between [self,{2,2}];
}
experiment validation type:gui {}
