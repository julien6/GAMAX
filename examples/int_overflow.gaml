model int_overflow

global {
    init { create counter number: 4; }
}

species counter {
    int value <- 2147483640 + index;
    int previous <- 0;
    reflex remember { previous <- value; }
    reflex increase { value <- value + 17; }
}
experiment validation type: gui {}
