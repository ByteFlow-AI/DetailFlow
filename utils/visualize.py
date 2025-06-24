from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image, text_list, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size=20, padding=15):

    width, height = image.size
    num_images = len(text_list)
    single_image_width = width // num_images

    offset = padding * 4
    new_height = height + font_size + offset
    new_image = Image.new('RGB', (width, new_height), (255, 255, 255))
    new_image.paste(image, (0, font_size + offset))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(new_image)

    for i, text in enumerate(text_list):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = i * single_image_width + (single_image_width - text_width) // 2
        text_y = padding
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    return new_image