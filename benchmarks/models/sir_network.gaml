// Undirected Watts-Strogatz network stored as two outgoing edge endpoints per node.
model sir_network

global {
    int ticks <- 0;
    int scenario <- 0;
    init { create person number: 32; }
    reflex capture { ticks <- ticks + 1; ask person { previous <- status; } }
}
species person {
    int link_a <- (index + 1) mod 32;
    int link_b <- (index + 2) mod 32;
    int status <- 0;
    int previous <- 0;
    int infection_age <- 0;
    int infected_contacts <- 0;
    // @initialization
    reflex disease {
        infected_contacts <- length(person where ((each.index = link_a or each.index = link_b or each.link_a = index or each.link_b = index) and each.previous = 1));
        if previous = 0 and infected_contacts > 0 { status <- 1; infection_age <- 0; }
        if previous = 1 {
            infection_age <- infection_age + 1;
            if infection_age >= 6 { status <- 2; }
        }
    }
}
experiment validation type: gui {}
