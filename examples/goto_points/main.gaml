model goto_points
global {
    geometry shape <- square(1000);
    init {create Walker number:3;}
}
species Walker skills:[moving] {
    float speed <- 15.0;
    point destination <- {0,0};
    int phase;
    reflex travel {
        if phase mod 5=0 {destination <- {rnd(999),rnd(999)};}
        do goto target:destination;
        phase <- phase+1;
    }
}
experiment validation type:gui {}
