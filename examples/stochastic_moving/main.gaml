model stochastic_moving
import "walker.gaml"
global { init { create walker number: 8; rng <- "java"; seed <- 17; } }
experiment validation type: gui {}
