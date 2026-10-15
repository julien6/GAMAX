model raster_matrix
global {
    image_file raster <- image_file('images/tiny.png');
    matrix<float> values <- matrix<float>(as_matrix(raster,{3,4}));
    init {
        float min <- min(values);
        float max <- max(values);
        float range <- (max-min)/2.5;
        loop i from: 0 to: 2 {
            loop j from: 0 to: 3 { values[i,j] <- (values[i,j]-min)/range; }
        }
    }
}
grid Cell width:1 height:1 frequency:0 {}
experiment validation type:gui {}
