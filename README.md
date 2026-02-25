# Denoising-AutoEncoder

Frameworks used: kivy, tensorflow/keras. Python version 3.7.

Input: 32×32×3

Encoder:
  Conv2D 32 → BatchNorm → Conv2D 32 → BatchNorm → Dropout
  Conv2D 64 → BatchNorm → Conv2D 64 → BatchNorm → Dropout
  Conv2D 128 → BatchNorm → Conv2D 128 → BatchNorm → Dropout
  Conv2D 256 → BatchNorm → Conv2D 256 → BatchNorm → Dropout
  Flatten
  Dense(latent_dim=256) → Latent vector

Decoder:
  Dense 2×2×256
  Reshape 2×2×256
  Decoder block 256 → Concatenate skip from Encoder
  Decoder block 128 → Concatenate skip
  Decoder block 64 → Concatenate skip
  Decoder block 32
  Conv2D 3 → Output reconstructed image (sigmoid)

Noise predictor:
  Dense 1 → Output predicted noise level

Examples:
<img width="798" height="520" alt="image" src="https://github.com/user-attachments/assets/05bdd28c-1fe6-4d49-af54-9d4d7c929bf0" />

<img width="801" height="526" alt="image" src="https://github.com/user-attachments/assets/243f6546-1958-45aa-b6b6-8cf4f8667fa8" />

<img width="798" height="524" alt="image" src="https://github.com/user-attachments/assets/b655b128-b464-4650-a6b5-7e8a708e67ba" />



