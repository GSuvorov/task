#Task-2
#Created by Vorobyev Ruslan
#2016
import glob
from scipy.ndimage.measurements import find_objects, label
from skimage import measure, exposure, filters
from numpy import where, shape, sort, zeros, percentile, array
from scipy import ndimage
from skimage.io import imread,imsave
from skimage.morphology import closing, rectangle
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.measure import regionprops
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk

#Функция превращает изображение в полутоновое
def rgb2gray (rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Функция меняет контраст изображения
def contrast (img, percent):

    dark = percentile(img, percent)
    bright = percentile(img, 100 - percent)
    shifted = img - dark
    dif = bright - dark
    img = shifted / dif
    img[where(img > 1)] = 1
    img[where(img < 0)] = 0
    return img

def main ():	
	
	#Считываем изображение и приводим все значения к отрезку [0,1]
	text=input('Введите название файла с картинкой (без .jpg):\n')
	f_name=text+'.jpg'
	img = imread(f_name)
	saved = img.copy()
	img = img.astype("float32")/255

	#Инициализируем переменные
	z=0
	delta=0
	percent=0
	a=len(img)
	b=len(img[0])

	#Запускаем цикл по нахождению метки документа, работаем пока не нашли
	while z==0:
		if percent!=0:
			img = contrast(img,percent)
			if percent>=5:
				img = ndimage.median_filter(img, 3)

		#Полутоновое изображение
		gray = rgb2gray(img) 

		#Бинаризация изображения
		gray[where(gray>=0.5)]=1
		gray[where(gray<0.5)]=0
		
		#Разметка найденных объектов
		label1,n = label(gray)

		#Получение данных для каждого из помеченных объектов
		cur=0
		props=regionprops(label1)
		m=zeros(len(props))

		#Цикл, который ограничивает возможность случая, когда объект == документ. Постепенно снижает свою требовательность
		for i in range(len(props)):
			
			t,y,r,w=props[i].bbox
			m[i]=(r-t)*(w-y)
			
			if props[i].convex_area<(0.3-delta)*a*b or props[i].convex_area>(0.7+delta)*a*b :
				continue

			if props[i].area > cur:
				cur=props[i].area
				z=i+1
				continue

		#Цикл, похожий на предыдущий, только с другими ограничениями. Так же постепенно снижает свою требовательность
		for i in range(len(props)):

			if m[i]<(0.3-delta)*a*b or m[i]>(0.7+delta)*a*b :
				continue

			if props[i].area > cur:
				cur=props[i].area
				z=i+1

		#Если пробежались по циклу много раз, во избежании зацикливания - случай останова цикла
		if delta!=0.25:
			delta+=0.05
			percent+=5
		else:
			z=1

	#Обрезаем наш документ
	t,y,r,w=props[z-1].bbox
	end = saved[t:r:1,y:w:1]

	#Попытка убрать ?шумы? увенчалась неуспехом...	
	#end = ndimage.median_filter(end, 3)

	#Сохраняем полученное изображение
	imsave('test.jpg',end)

main()