const position_images = ()=>{
  let images = $('.img');

  images.map((idx, img) => {
    let image = $(img);
    let assign_to = image.attr('assign_to')

    // get location of assign to
    let assign_to_element = $('#' + assign_to);
    let offset = assign_to_element.offset();
    let width = assign_to_element.width();

    image.css('left', offset.left + width + 50 + 'px');
    image.css('top', offset.top + 'px');
  })
}

$(document).ready(position_images);
setInterval(position_images, 2000);
$(window).resize(position_images);