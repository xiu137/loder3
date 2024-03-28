import main2,os

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.makedirs('output')
    input_dir = r'data/'
    for file in os.listdir(input_dir):
        main2.main(
            input_img=input_dir+file,
            output_img="./output/img/"+file,
            output_label="./output/label/"+file,
            model_path=r'./best (2).pt',
            size_stride=[(4500,900),(3000,600),(2000,400)],
            confidence=0.6,
            exclude=0.15)