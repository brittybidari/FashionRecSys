import tensorflow as tf
import os
import argparse
import json

def custom_fashion_recommendation_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # First Convolutional Layer with more filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Second Convolutional Layer with more filters
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Third Convolutional Layer with more filters
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # Flatten the features for the fully connected layers
        tf.keras.layers.Flatten(),

        # First Fully Connected Layer with more units
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Second Fully Connected Layer with more units
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--validation", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--input_shape", type=str, default="[28, 28, 1]")  # Provide the default input shape as a string
    return parser.parse_args()

def main():
    args = parse_args()

    # Convert the input shape argument from a string to a list of integers
    input_shape = json.loads(args.input_shape)

    # Load data using ImageDataGenerator
    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    train_generator = train_data_gen.flow_from_directory(args.train, target_size=(input_shape[0], input_shape[1]), batch_size=args.batch_size, class_mode='categorical')

    val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    val_generator = val_data_gen.flow_from_directory(args.validation, target_size=(input_shape[0], input_shape[1]), batch_size=args.batch_size, class_mode='categorical')

    # Get number of classes from the data generator
    num_classes = train_generator.num_classes

    # Create the model
    model = custom_fashion_recommendation_model(input_shape, num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=args.epochs, validation_data=val_generator)

    # Save the trained model
    model.save(os.path.join(args.model_dir, 'fashion_recommendation_model'))

if __name__ == '__main__':
    main()
