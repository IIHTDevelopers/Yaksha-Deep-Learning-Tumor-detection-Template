#pip install requirements.text
import data
import model
import train
import predict

def main():
    while True:
        choice = input("Choose an option (train/predict/exit): ").strip().lower()

        if choice == "train":
            data_dir = "CVC-612/"
            # Ask the user for training parameters
            print("batch size, lr and epochs decide accuracy and iou values")
            batch_size = int(input("Enter batch size: default is 8 "))
            lr = float(input("Enter learning rate: default is 1e-4 "))
            epochs = int(input("Enter the number of epochs:default is 20 "))
            # batch_size = 8
            # lr = 1e-4
            # epochs = 20
            # Print data details
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data.load_data(data_dir)
            print("Data Details:")
            print(f"Number of training samples: {len(train_x)}")
            print(f"Number of validation samples: {len(valid_x)}")
            print(f"Number of test samples: {len(test_x)}")
            # Create the model instance for model summary
            model_instance = model.build_model()
            # Print model summary
            print("Model Summary:")
            model_instance.summary()
            # Train the model
            train.train_unet(data_dir, batch_size, lr, epochs)
            print("Training completed.")
        elif choice == "predict":
            data_dir = "test-images/"
            predict.make_predictions(data_dir)
            print("Prediction completed.")
        elif choice == "exit":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 'train', 'predict', or 'exit'.")

if __name__ == "__main__":
    main()
