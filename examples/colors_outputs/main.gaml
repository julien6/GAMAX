model colors_outputs

global {
    rgb color <- #white;
    int matches;
    float value <- 0.25;
    init {
        create walker number: 3;
        write "Initial value: " + value;
    }
    reflex update {
        color <- rgb(46,204,113);
        matches <- walker count (each.color = color);
    }
}

species walker {
    rgb color <- rgb(46,204,113);
    reflex update { color <- #red; }
}

experiment validation type: gui {}
