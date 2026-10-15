model grid_species_order
global {
    geometry shape <- rectangle(7,5);
    init {create Walker number:2 with:(location:{3.5,2.5});}
    reflex queue {diffuse var:road on:Cell proportion:1.0 radius:3 propagation:gradient method:convolution;}
}
grid Cell width:7 height:5 neighbors:8 {
    float road <- 0.0 update: road<=5.0 ? 0.0 : road-5.0;
    rgb color <- rgb(0,99,0) update:rgb(0,99,0)+int(road*5);
}
species Walker skills:[moving] {
    float observed;
    reflex deposit {observed <- Cell(location).road;Cell(location).road <- Cell(location).road+100.0;}
}
experiment validation type:gui {}
