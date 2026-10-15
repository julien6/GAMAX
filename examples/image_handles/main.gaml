model image_handles
global {
    image_file first_image <- image_file('images/first.png');
    image_file second_image <- image_file('images/second.png');
    list<image_file> images <- [first_image,second_image];
    init {create Walker number:3;}
}
species Walker {
    image_file icon;
    reflex pick {if flip(0.5) {icon <- first_image;} else {icon <- second_image;}}
}
experiment validation type:gui {}
