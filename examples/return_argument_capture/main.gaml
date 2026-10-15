model return_argument_capture
global {int value <- 0; int total <- 0;
    int adjust(int input) {int local <- input+1; return local*2;}
    reflex execute {int argument <- value+3;value <- value+1; total <- total+adjust(argument);}
}
experiment validation type:gui {}
