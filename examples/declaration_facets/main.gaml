model declaration_facets

global {
    int lower <- 2;
    int upper <- 7;
    int bounded <- 0 min: lower max: upper step: 2;
    int inconsistent <- 0 min: 9 max: 3;
    float continuous <- 20.5 min: -1.5 max: 5.5 step: 0.3;
    int zero;
    float zero_float;
    bool disabled;
    init { create walker number: 4; }
    reflex constraints {
        bounded <- bounded + 3;
        continuous <- continuous - 3.0;
        inconsistent <- -1;
    }
}

species walker {
    int value <- index min: 1 max: 2 step: 2;
    int untouched;
    reflex update { value <- value + 1; }
}

experiment validation type: gui {}
