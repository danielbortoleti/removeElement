import cv2
import numpy as np
import matplotlib.pylab as plt

aquario = cv2.imread("imagens/coragem-casa.jpg")
imagem = cv2.imread("imagens/coragem.jpg")

##Converter para HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

##Criar a mascara do laranja
##Vermelho (5, 75, 25)
##Laranja (25, 255, 255)
mascara = cv2.inRange(imagem_hsv, (90,0,0), (110,255,255))
imascara = mascara>0

peixe = np.zeros_like(imagem, np.uint8)
peixe[imascara] = imagem[imascara]

peixe_amarelo = imagem.copy()
imagem_hsv[...,0] = imagem_hsv[...,0] + 20
peixe_amarelo[imascara] = cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2BGR)[imascara]
peixe_amarelo = np.clip(peixe_amarelo, 0, 255)

aquario_peixe = cv2.bitwise_and(aquario, aquario, mask=imascara.astype(np.uint8))
sem_peixe = cv2.bitwise_and(imagem, imagem, mask=(np.bitwise_not(imascara).astype(np.uint8)))
sem_peixe = sem_peixe + aquario_peixe

plt.figure(figsize=(20, 12))
plt.subplot(221), plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.subplot(222), plt.imshow(cv2.cvtColor(aquario, cv2.COLOR_BGR2RGB))
plt.subplot(223), plt.imshow(cv2.cvtColor(peixe_amarelo, cv2.COLOR_BGR2RGB))
plt.subplot(224), plt.imshow(cv2.cvtColor(sem_peixe, cv2.COLOR_BGR2RGB))
plt.show()

