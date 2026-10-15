model moving_defaults global {int draw; init {create Walker number:3;} reflex sample {draw <- rnd(100);}} species Walker skills:[moving] {int sample <- rnd(100);} experiment validation type:gui {}
