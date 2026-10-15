model native_location_setter
global { init { create walker number: 8; } }
species walker skills: [moving] {
    init {
        location <- {20.0 + index, 20.0 + index};
        speed <- 0.005;
        heading <- index * 45.0;
    }
    reflex reposition {
        location <- location + {cos(heading) + 1.0, sin(heading) + 1.0};
        do move;
    }
}
experiment validation type: gui {}
