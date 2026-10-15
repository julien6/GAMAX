model creation_captures_death
global {
    int completed;
    int observed_population;
    init {
        int offset <- 10;
        point origin <- {2,2};
        create Walker number:4 {
            if index=1 {do die;}
            value <- value+offset;
            location <- origin+{index,0};
            completed <- completed+1;
        }
        observed_population <- length(Walker);
    }
}
species Walker {float value <- rnd(1.0);reflex advance {value <- value+1.0;}}
experiment validation type:gui {}
