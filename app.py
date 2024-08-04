import omr_processing


if __name__ == "__main__":
    image_path = r"final_template_omr_filled.png"
    omr_processing.process_image(image_path)

    dict = omr_processing.answer

    print("---------------------------------")
    for x in dict:
        print(x, dict[x])
