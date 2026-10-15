model parameter_grid
global {
    int columns <- 4 min:2;
    int rows <- 3;
    geometry shape <- square(12);
    int draw;
    reflex sample {draw <- rnd(100);}
}
grid cell width:columns height:rows neighbors:8 frequency:0 use_regular_agents:false {
    int sample <- rnd(100);
}
experiment validation type:gui {}
