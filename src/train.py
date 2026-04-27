from dataset import load_dataset
from face_auth import FaceAuthenticator

def main():
    print("Loading dataset...")
    images, labels, label_map = load_dataset("data/enrolled")
    print(f"Loaded {len(images)} images")
    model = FaceAuthenticator()
    print("Training model...")
    model.train(images, labels, label_map)
    print("Saving model...")
    model.save()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()