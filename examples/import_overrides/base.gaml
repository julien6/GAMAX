model import_base
global {
    int value <- 2;
    int signature;
    init {signature <- signature*10+1;}
    action hook virtual:true;
    action compute {signature <- signature*10+value; do hook;}
    reflex tick {signature <- signature+value;}
}
species Walker {int value <- 4;}
