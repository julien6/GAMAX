model scheduling_creation_order

global {
    int signature;
    init {create Second number:1; create First number:1;}
    reflex clear {signature <- 0;}
}
species First {reflex update {signature <- signature*10+1;}}
species Second {reflex update {signature <- signature*10+2;}}
grid cell width:2 height:2 use_regular_agents:false {
    int value;
    reflex update {signature <- signature*10+3; value <- value+1;}
}
experiment validation type:gui {}
