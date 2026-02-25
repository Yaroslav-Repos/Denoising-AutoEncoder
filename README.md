# Denoising-AutoEncoder

Frameworks used: kivy, tensorflow/keras. Python version 3.7.

Input: 32×32×3

Encoder:<br>
  Conv2D 32 → BatchNorm → Conv2D 32 → BatchNorm → Dropout<br>
  Conv2D 64 → BatchNorm → Conv2D 64 → BatchNorm → Dropout<br>
  Conv2D 128 → BatchNorm → Conv2D 128 → BatchNorm → Dropout<br>
  Conv2D 256 → BatchNorm → Conv2D 256 → BatchNorm → Dropout<br>
  Flatten<br>
  Dense(latent_dim=256) → Latent vector<br>

Decoder:<br>
  Dense 2×2×256<br>
  Reshape 2×2×256<br>
  Decoder block 256 → Concatenate skip from Encoder<br>
  Decoder block 128 → Concatenate skip<br>
  Decoder block 64 → Concatenate skip<br>
  Decoder block 32<br>
  Conv2D 3 → Output reconstructed image (sigmoid)<br>

Noise predictor:<br>
  Dense 1 → Output predicted noise level<br>

Examples:
<img width="798" height="520" alt="image" src="https://github.com/user-attachments/assets/05bdd28c-1fe6-4d49-af54-9d4d7c929bf0" />

<img width="801" height="526" alt="image" src="https://github.com/user-attachments/assets/243f6546-1958-45aa-b6b6-8cf4f8667fa8" />

<img width="798" height="524" alt="image" src="https://github.com/user-attachments/assets/b655b128-b464-4650-a6b5-7e8a708e67ba" />

<img width="801" height="528" alt="image" src="https://github.com/user-attachments/assets/5a62fa01-b311-4d7f-9b21-8ea7c9213dc1" />

<img width="799" height="527" alt="image" src="https://github.com/user-attachments/assets/807fceaa-eb78-4d31-bd13-94921928584b" />





