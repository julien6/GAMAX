model nullable_colors
global {
    rgb missing;
    rgb explicit_nil <- nil;
    rgb chosen;
    bool is_nil;
    init {create Walker number:2;}
    reflex check {is_nil <- missing = nil; chosen <- is_nil ? explicit_nil : #red;}
}
species Walker {
    rgb color;
    bool is_nil;
    reflex toggle {
        if flip(0.5) {color <- nil;} else {color <- #red;}
        is_nil <- color = nil;
    }
}
experiment validation type:gui {}
