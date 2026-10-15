model wander_rules
global {
    float amplitude <- 45.0;
    float distance <- 20.0;
    geometry shape <- square(100);
    init {create Walker number:3;}
}
species Walker skills:[moving] {
    bool branch;
    reflex walk {
        branch <- flip(0.5);
        if branch {do wander amplitude:amplitude speed:distance;}
        else {do move heading:heading+15.0 speed:5.0;}
    }
}
experiment validation type:gui {}
