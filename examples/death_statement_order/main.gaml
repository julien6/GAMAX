model death_statement_order
global {int count; init {create Walker number:3;}}
species Walker {
    reflex remove when:index=1 {do die; count <- count+100;}
    reflex update {count <- count+1;}
}
experiment validation type:gui {}
