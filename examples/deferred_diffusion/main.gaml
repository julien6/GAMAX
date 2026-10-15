model deferred_diffusion
global {
    geometry shape <- rectangle(7,5);
    int tick <- 0;
    init {ask Cell {road <- index=17 ? 100.0 : (index=0 ? -1.0 : 0.0);}}
    reflex diffuse_before_deposit {
        diffuse var:road on:Cell proportion:0.5 radius:3 propagation:gradient method:convolution;
        tick <- tick+1;
        Cell({0.5,0.5}).road <- tick*10.0;
    }
}
grid Cell width:7 height:5 neighbors:8 {
    float road <- 0.0;
    reflex decay {road <- road-0.5;}
}
experiment validation type:gui {}
