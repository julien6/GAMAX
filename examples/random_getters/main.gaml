model random_getters
global {
    int next;
    int random_value -> rnd(100);
    init {create Walker number:3;}
    reflex sample {next <- rnd(100);}
}
species Walker {
    int value <- rnd(100);
    int random_value -> rnd(100);
    int derived -> random_value+rnd(100);
    reflex sample {value <- rnd(100);}
}
experiment validation type:gui {}
