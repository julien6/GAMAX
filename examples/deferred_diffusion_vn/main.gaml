model deferred_diffusion_vn
global {
    geometry shape <- rectangle(4,3);
    int tick <- 0;
    init {create Walker number:1 with:(location:{3.5,2.5});}
    reflex queue {
        diffuse var:road on:Cell proportion:(tick mod 2=0 ? 0.0 : 0.25) radius:1 propagation:gradient method:convolution;
        tick <- tick+1;
    }
}
grid Cell width:4 height:3 neighbors:4 frequency:2 {
    float road <- 0.0 update:road<=3.0 ? 0.0 : road-3.0;
}
species Walker skills:[moving] {reflex deposit {Cell(location).road <- Cell(location).road+100.0;}}
experiment validation type:gui {}
