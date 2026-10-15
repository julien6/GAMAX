model initialization

global {
    int base <- 2;
    init {
        base <- 5;
        create cohort_a number: 3;
        base <- 7;
        create cohort_b number: 2;
    }
    reflex advance { base <- base + 1; }
}

species cohort_a {
    int value <- base + index;
    int previous <- 0;
    init { value <- value * 2; }
    reflex remember { previous <- value; }
    reflex update { value <- value + base; }
}

species cohort_b {
    int value <- base + index;
    int previous <- 0;
    reflex remember { previous <- value; }
    reflex update {
        if value < 20 { value <- value + base; }
        else { value <- value - 3; }
    }
}
experiment validation type: gui {}
