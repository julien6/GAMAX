model list_aliases
global {
    list<Walker> first_list;
    list<Walker> second_list;
    list<Walker> third_list;
    int phase;
    init {
        create Walker number:3;
        first_list <- Walker as list;
        second_list <- first_list;
        third_list <- first_list;
    }
    reflex edit {
        if phase=0 {second_list >> (first_list at 0);}
        else if phase=1 {second_list <- shuffle(Walker);}
        else if phase=2 {third_list << (Walker at 0);}
        phase <- phase+1;
    }
}
species Walker {int value <- index;}
experiment validation type:gui {}
