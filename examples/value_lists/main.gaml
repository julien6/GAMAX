model value_lists

global {
    rgb first_color <- rgb("orange");
    rgb second_color <- rgb("green");
    rgb third_color <- #pink;
    list colors <- [first_color,second_color,third_color] of:rgb;
    list<int> numbers <- [4,2,7];
    list<point> places <- [{1,2},{3,4},{5,6}];
    list<bool> flags <- [true,false,true];
    int selected;
    rgb chosen <- #black;
    int result;
    point destination <- {0,0};
    bool flag;
    init {create Walker number:1;}
    reflex choose {
        selected <- rnd(2);
        chosen <- colors at selected;
        result <- numbers at selected;
        destination <- places at selected;
        flag <- flags at selected;
    }
}
species Walker {rgb chosen <- colors at rnd(2);}
experiment validation type:gui {}
