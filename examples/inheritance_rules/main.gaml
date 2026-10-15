model inheritance_rules

global {
    int init_calls;
    int draw;
    init { create Child number: 2; }
    reflex draw_next { draw <- rnd(100); }
}
species Parent {
    int a <- rnd(100);
    int b <- rnd(100);
    init { init_calls <- init_calls + 1; }
    reflex update { a <- a + 1; }
}
species Child parent: Parent {
    int b <- rnd(200);
    int c <- rnd(100);
    init { init_calls <- init_calls + 10; }
    reflex update { a <- a + 10; }
}
experiment validation type: gui {}
