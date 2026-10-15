model computed_attributes
global {
    int a <- 1 update:a+1;
    int doubled -> a*2;
    int tripled -> {doubled+a};
    int selected;
    init {create Walker number:2;}
    reflex mark {selected <- tripled;}
}
species Walker {
    int value <- 4 update:value+1;
    int derived -> value+doubled;
    bool high -> derived>10;
    int selected;
    reflex mark {selected <- derived;}
}
experiment validation type:gui {}
