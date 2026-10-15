model update_forward_dependencies
global {
    int b <- 2 update:a+10;
    int a <- 1 update:a+1;
    int signature;
    init {create Walker number:2;}
    reflex mark {signature <- a*100+b;}
}
species Walker {
    int y <- 2 update:x+10;
    int x <- 1 update:x+a;
    int marker;
    reflex mark {marker <- x*100+y;}
}
experiment validation type:gui {}
