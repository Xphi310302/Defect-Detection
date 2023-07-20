from Vanila_Unet_model import *
import process_image as process
import matplotlib.pyplot as plt
path = '0b31575c2.jpg'
unet = Vanila_Unet().model_gen()
# print(unet.summary())
pred_mask = process.predict(path, unet, False)
fig, axs = plt.subplots(2,1, figsize=(6, 3))
# axs[0].imshow(true_mask)
# axs[0].axis('off')
# axs[0].set_title(f'{name}')
axs[1].imshow(pred_mask)
axs[1].axis('off')
print(pred_mask.shape)
print(type(pred_mask))
plt.show()