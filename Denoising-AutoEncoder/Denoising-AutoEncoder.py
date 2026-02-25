import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Крок 1: Завантаження, нормалізація і збереження датасету
def prepare_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)

# Крок 2: Завантаження даних з кешу
def load_data():
    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")
    return x_train, x_test

# Крок 3: Функція додавання шуму з відомим рівнем
def add_noise_with_known_level(images, min_noise=0.0, max_noise=0.5):
    noise_factors = np.random.uniform(min_noise, max_noise, size=(images.shape[0], 1))
    noise_factors = noise_factors.reshape(-1, 1, 1, 1)  # правильна форма
    noise = noise_factors * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images, noise_factors.reshape(-1, 1)

# Крок 4: Побудова автоенкодера: енкодера, латентного простору та декодера
def encoder_block(x, filters, strides=2):
    x = layers.Conv2D(filters, (3, 3), strides = 1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), strides = strides, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    return x

def decoder_block(x, filters):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    return x

def build_autoencoder(latent_dim = 256):
    input_img = tf.keras.Input(shape=(32, 32, 3), name="image_input") #Вхід 32×32×3 = 3072

    # Енкодер (3 рівні)
    x1 = encoder_block(input_img, 32)   # 32×32 → 16×16 → 16×16×32 = 8192
    x2 = encoder_block(x1, 64)          # 16×16 → 8×8 → 8×8×64 = 4096
    x3 = encoder_block(x2, 128)         # 8×8 → 4×4 → 4×4×128 = 2048
    x4 = encoder_block(x3, 256)         # 4×4 → 2×2 → 2×2×256 = 1024

    x_flat = layers.Flatten()(x4)       #2×2×256 = 1024

    # Латентний простір за замовчуванням = 256 < 3072
    latent = layers.Dense(latent_dim, activation="relu", name="latent_vector")(x_flat)

    # Декодер
    x = layers.Dense(2*2*256, activation="relu")(latent) #2×2×256 = 1024

    x = layers.Reshape((2, 2, 256))(x)
    x = decoder_block(x, 256)  # 2×2 → 4×4
    x = layers.Concatenate()([x, x3])  # skip з 4×4

    x = decoder_block(x, 128)  # 4×4 → 8×8
    x = layers.Concatenate()([x, x2])  # skip з 8×8

    x = decoder_block(x, 64)   # 8×8 → 16×16
    x = layers.Concatenate()([x, x1])  # skip з 16×16

    x = decoder_block(x, 32)   # 16×16 → 32×32

    output_img = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="reconstructed_image")(x)

    noise_pred = layers.Dense(1, activation="linear", name="noise_level_pred")(latent)

    model = tf.keras.Model(inputs=input_img, outputs=[output_img, noise_pred])

    model.compile(
        optimizer="adam",
        loss={
            "reconstructed_image": "mse",
            "noise_level_pred": "mse"
        },
        loss_weights={
            "reconstructed_image": 1.0,
            "noise_level_pred": 0.1
        },
        metrics={
            "reconstructed_image": ["mae"],
            "noise_level_pred": ["mae"]
        }
    )

    model.summary()
    return model

# Крок 5: Візуалізація результатів
def plot_results(x_test, noisy_imgs, decoded_imgs, true_noise_levels, pred_noise_levels, n=10):
    plt.figure(figsize=(20, 8))
    for i in range(n):
        # Оригінал
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("Оригінал")
        plt.axis("off")

        # Зашумлене
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(noisy_imgs[i])
        plt.title("З шумом")
        plt.axis("off")

        # Вивід рівня шуму під зашумленим зображенням
        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.text(0.5, 0.5,
                 f"Рівень шуму: {true_noise_levels[i][0]:.3f}\nПередбачений:{pred_noise_levels[i][0]:.3f}",
                 fontsize=10,
                 ha='center', va='center')
        plt.axis("off")

        # Відновлене
        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(decoded_imgs[i])
        plt.title("Відновлене")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    # Втрата для реконструкції
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['reconstructed_image_loss'], label='train loss (reconstruction)')
    plt.plot(history.history['val_reconstructed_image_loss'], label='val loss (reconstruction)')
    plt.xlabel('Епоха')
    plt.ylabel('MSE Loss')
    plt.title('Втрата реконструкції')
    plt.legend()
    
    # Метрика MAE для реконструкції
    plt.subplot(1, 2, 2)
    plt.plot(history.history['reconstructed_image_mae'], label='train MAE (reconstruction)')
    plt.plot(history.history['val_reconstructed_image_mae'], label='val MAE (reconstruction)')
    plt.xlabel('Епоха')
    plt.ylabel('MAE')
    plt.title('MAE реконструкції')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Основна функція
def main():
    if not os.path.exists("x_train.npy") and not os.path.exists("x_test.npy"):
        prepare_data()

    x_train, x_test = load_data()

    # Генерація шуму з відомими коефіцієнтами
    x_train_noisy, noise_train = add_noise_with_known_level(x_train)
    x_test_noisy, noise_test = add_noise_with_known_level(x_test)

    autoencoder = build_autoencoder()

    history = autoencoder.fit(
        x_train_noisy,
        {"reconstructed_image": x_train, "noise_level_pred": noise_train},
        validation_data=(x_test_noisy, {"reconstructed_image": x_test, "noise_level_pred": noise_test}),
        epochs=25,
        batch_size=128,
        shuffle=True
    )

    decoded_imgs, predicted_noise = autoencoder.predict(x_test_noisy)

    plot_results(x_test, x_test_noisy, decoded_imgs, noise_test, predicted_noise, n=10)

    plot_training_history(history)

    autoencoder.save("autoencoder_v2.h5")

if __name__ == "__main__":
    main()

