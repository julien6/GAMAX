model goto_agents
global {geometry shape <- rectangle(500,300); init {create Target number:4;create Walker number:2;}}
species Target skills:[moving] {reflex walk {do wander amplitude:45.0 speed:1.0;}}
species Walker skills:[moving] {
    float following <- 0.0;
    action seek_agent(Target other) {do goto target:other speed:2.0;}
    reflex advance {do seek_agent(one_of(Target));following <- rnd(1.0);}
}
experiment validation type:gui {}
