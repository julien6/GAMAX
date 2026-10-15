model raster_pixels
global {
    file raster <- image_file('images/tiny.png');
}
grid Cell width: 2 height: 2 neighbors: 4 {
    int pixel <- raster at {grid_x,grid_y};
    rgb color <- rgb(pixel);
    int red <- (color as list) at 0;
    int alpha <- (color as list) at 3;
}
experiment validation type:gui {}
