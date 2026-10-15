model pure_getter_maps
global {
 int walkers <- 600;
 float perception <- 8.0;
 int group_total <- 0;
 int neighbor_total <- 0;
 int happy_total <- 0;
 int draw_after_maps <- 0;
 init {create Walker number:walkers;}
 reflex vary {perception <- (cycle mod 2 = 0) ? 8.0 : 12.0;}
 reflex measure {
  group_total <- sum(Walker collect each.same_group);
  neighbor_total <- sum(Walker collect each.total_nearby);
  happy_total <- Walker count each.is_happy;
  draw_after_maps <- rnd(1000);
 }
 reflex spawn when: cycle = 4 {create Walker number:2;}
}
species Walker {
 int group <- rnd(1);
 list<Walker> nearby -> Walker at_distance perception;
 int total_nearby -> length(nearby);
 int same_group -> nearby count (each.group = group);
 bool is_happy -> same_group >= total_nearby / 2;
 reflex mutate {
  if cycle mod 2 = 0 {location <- {rnd(100.0),rnd(100.0)};}
  if cycle = 3 and index = 0 {do die;}
 }
}
experiment validation type:gui {output {}}
