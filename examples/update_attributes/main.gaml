model update_attributes
global {
    int a <- 1 update:a+1;
    int b <- 2 update:a+10;
    int signature;
    init {create Walker number:2;}
    reflex mark {signature <- a*100+b;}
}
species Walker {
    int x <- 1 update:x+a max:10000;
    float energy <- 2.5 update:energy-0.125 min:0.0;
    int marker;
    reflex mark {marker <- x;}
}
grid cell width:2 height:2 frequency:1 use_regular_agents:false {
    int x <- 1 update:x+a;
    int marker;
    reflex mark {marker <- x;}
}
experiment validation type:gui {}
