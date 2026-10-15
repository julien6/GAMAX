model nested_asks
global {
    init {
        create Sender number:2;
        create Receiver number:3;
        ask Receiver {value <- index*2;}
    }
}
species Sender {
    int value <- index+1;
    point displacement <- {0,0};
    reflex interact {
        point accumulated <- {0,0};
        ask (Receiver where (each.value>=0)) {
            accumulated <- accumulated+location-myself.location;
            ask (Sender where (each.index = myself.index mod 2)) {value <- value+1;}
            value <- value+myself.value;
        }
        displacement <- accumulated;
    }
}
species Receiver {
    int value <- rnd(100);
}
experiment validation type:gui {}
