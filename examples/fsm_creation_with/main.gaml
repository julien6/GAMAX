model fsm_creation_with
global {
    init { create Walker number:3 with:(location:{50,50}); }
}
species Walker skills:[moving] control:fsm {
    int value <- 0;
    int marker <- 0;
    reflex count { value <- value+1; marker <- rnd(0,100); }
    state searching initial:true {
        value <- value+2;
        transition to:carrying when:value>10;
        transition to:following when:true;
    }
    state carrying {
        value <- value+5;
        transition to:searching when:true;
    }
    state following {
        value <- value+7;
        transition to:searching when:true;
    }
}
experiment validation type:gui {}
