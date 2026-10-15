model guard_probe
global {int total <- 0; init {create Actor number:4;}}
species Actor {int value <- 0; reflex change when: cycle mod 2 = 0 {int local_draw <- rnd(10); value <- value + local_draw; total <- total + 1;} reflex skipped when: cycle < 0 { value <- value + rnd(1000); }}
experiment validation type:gui {output {}}
