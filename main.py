import argparse

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Unet Fine-Grained SBIR model')
    parsers.add_argument('--is_train_unet', type=bool, default=True)