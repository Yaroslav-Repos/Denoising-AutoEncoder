import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from tensorflow.keras.datasets import cifar10
from PIL import Image as PILImage
import os
from tensorflow.keras.models import load_model
from kivy.uix.label import Label 

Window.size = (800, 400)


def array_to_texture(img_array):
    img_array = (img_array * 255).astype(np.uint8)
    pil_img = PILImage.fromarray(img_array)
    pil_img = pil_img.resize((256, 256))
    data = pil_img.tobytes()
    texture = Texture.create(size=pil_img.size, colorfmt='rgb')
    texture.blit_buffer(data, colorfmt='rgb', bufferfmt='ubyte')
    return texture

def prepare_data():
    _ , (x_test, _) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    np.save("x_test.npy", x_test)

def load_data():
    x_test = np.load("x_test.npy")
    return x_test

class DenoiseApp(App):
    def build(self):
        self.model = load_model('autoencoder_v1.h5')

        if not os.path.exists("x_test.npy"):
            prepare_data()

        self.x_test = load_data()


        self.current_index = None
        self.current_noisy = None

        layout = BoxLayout(orientation='horizontal', spacing=10, padding=10)

        self.left_box = BoxLayout(orientation='vertical')
        self.noisy_image = Image()
        self.noise_level_label = Label(text='Рівень шуму: 0%', size_hint=(1, 0.1))
        self.noise_button = Button(text='Випадкове зображення\n+\nвипадковий шум (від 0% до 50%)', size_hint=(1, 0.2))
        self.noise_button.bind(on_press=self.add_noise)

        self.left_box.add_widget(self.noisy_image)
        self.left_box.add_widget(self.noise_level_label)
        self.left_box.add_widget(self.noise_button)

        self.right_box = BoxLayout(orientation='vertical')
        self.denoised_image = Image()
        self.denoise_button = Button(text='Знешумити', size_hint=(1, 0.2))
        self.denoise_button.bind(on_press=self.denoise)

        self.right_box.add_widget(self.denoised_image)
        self.right_box.add_widget(self.denoise_button)

        layout.add_widget(self.left_box)
        layout.add_widget(self.right_box)

        return layout

    def add_noise(self, instance):
        idx = np.random.randint(0, len(self.x_test))
        self.current_index = idx
        clean_img = self.x_test[idx]
        noise_level = np.random.uniform(0.0, 0.5)
        noisy_img = clean_img + noise_level * np.random.normal(loc=0.0, scale=1.0, size=clean_img.shape)
        noisy_img = np.clip(noisy_img, 0., 1.)
        self.current_noisy = noisy_img

        self.noisy_image.texture = array_to_texture(noisy_img)

        self.noise_level_label.text = f'Рівень шуму: {int(noise_level * 100)}%'

    def denoise(self, instance):
        if self.current_noisy is None:
            return
        input_img = np.expand_dims(self.current_noisy, axis=0)
        denoised_img, _ = self.model.predict(input_img)
        denoised_img = denoised_img[0]
        self.denoised_image.texture = array_to_texture(denoised_img)


if __name__ == '__main__':
    DenoiseApp().run()
