model contact_probe
global {int walkers <- 600; float perception <- 8.0; int checksum <- 0; init {create Walker number:walkers;} reflex vary {perception <- (cycle mod 2 = 0) ? 8.0 : 12.0;} reflex spawn when: cycle = 4 {create Walker number:2;}}
species Walker {int group <- rnd(1); list<Walker> nearby -> Walker at_distance perception; int total_nearby -> length(nearby); int same_group -> nearby count (each.group = group); reflex mutate {if cycle mod 2 = 0 {location <- {rnd(100.0),rnd(100.0)};} checksum <- checksum + same_group; if cycle = 3 and index = 0 {do die;}}}
experiment validation type:gui {}
