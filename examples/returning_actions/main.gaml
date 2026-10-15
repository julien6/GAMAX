model returning_actions

global {
    int calls <- 0;
    init { create Walker number: 3; }
}
species Walker {
    int value <- 0;
    int marker <- 0;
    int choose(int threshold) {
        calls <- calls + 1;
        int draw <- rnd(0, 9);
        if draw < threshold { return draw; }
        value <- value + 1;
        return draw + 10;
        calls <- calls + 1000;
    }
    int compose { return choose(5) + choose(3); }
    reflex tick {
        marker <- compose();
        value <- value + 2;
    }
}
experiment validation type: gui {}
